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
    /// A paged file-space strategy was requested with a page size the writer
    /// cannot use: it must be a power of two of at least 512 bytes.
    InvalidFileSpacePageSize(u64),
    /// A paged file-space strategy was requested alongside a userblock that is
    /// not a whole number of pages. File-space pages are measured from the file
    /// base, so the two boundaries coincide only when the userblock divides by
    /// the page size: `(userblock bytes, page size)`.
    UserblockNotPageAligned(u64, u64),
    /// A userblock size the format does not define: it must be zero, or a power
    /// of two of at least 512 bytes. A reader looks for the superblock at 0, 512,
    /// 1024, and so on doubling, so any other size produces a file nothing can
    /// open.
    InvalidUserblockSize(u64),
    /// More userblock content was supplied than the userblock region holds. The
    /// overflow would displace the superblock, so it is refused rather than
    /// truncated.
    UserblockContentTooLarge {
        /// Bytes supplied.
        content: u64,
        /// Bytes the userblock region holds.
        userblock: u64,
    },
    /// A free-space manager block (`FSHD`/`FSSE`) is malformed.
    InvalidFreeSpaceManager,
    /// An enumeration datatype was built over a base type that is not an
    /// integer. HDF5 enumerations must have a fixed-point base.
    EnumBaseNotInteger,
    /// An enumeration member's value does not occupy exactly the base type's
    /// size: `(member name, expected bytes, actual bytes)`.
    EnumMemberValueSize(String, u32, usize),
    /// An enumeration member's integer value does not fit in the base type:
    /// `(member name, value, base size in bytes)`.
    EnumMemberValueRange(String, i64, u32),
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
    /// A fractal heap's huge-objects v2 B-tree is not the indirectly accessed,
    /// non-filtered layout (record type 1) this reader decodes: either the tree
    /// declares a different record type, or its records are too short to hold
    /// that one. Reading its records as that layout would decode an object ID
    /// out of another field's bytes.
    UnexpectedHugeObjectBTree {
        /// The record type the B-tree declares.
        tree_type: u8,
        /// The record size the B-tree declares, in bytes.
        record_size: usize,
        /// The bytes a type-1 record needs: address + length + object ID.
        required: usize,
    },
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
    /// Filter processing error, including a stream that failed to decode. Every
    /// filter in the pipeline — shuffle, scale-offset, LZF, deflate — reports a
    /// bad chunk with this variant, so "this chunk did not decode" is one match
    /// arm rather than one per compressor. The payload names the filter.
    FilterError(String),
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
    /// admit no format this crate writes. It writes the 1.8 and 1.10 formats
    /// ([`LibVer::WRITER_OLDEST`](crate::LibVer::WRITER_OLDEST) through
    /// [`LibVer::WRITER_DEFAULT`](crate::LibVer::WRITER_DEFAULT)), so an upper
    /// bound older than 1.8, or a lower bound newer than 1.10, is unsatisfiable.
    /// The fields carry the default format and the bounds asked for, as
    /// [`LibVer::name`](crate::LibVer::name) labels.
    LibverBoundsUnsatisfiable {
        /// The library-version label of the format this crate writes by default.
        writes: &'static str,
        /// The requested lower bound.
        requested_low: &'static str,
        /// The requested upper bound.
        requested_high: &'static str,
    },
    /// The file's content needs a newer on-disk format than the requested
    /// library-version bounds allow, so writing it would silently produce a file
    /// the caller asked not to receive.
    ///
    /// Reported rather than upgraded, because the bound exists to be relied on:
    /// a MAT v7.3 file bounded to 1.8 so MATLAB can load it is worth less than
    /// nothing if compressing it quietly restores the 1.10 format. `content`
    /// names what forced the newer format and `needs` the format it forces, both
    /// as [`LibVer::name`](crate::LibVer::name) labels.
    LibverTooOldForContent {
        /// What in the file requires a newer format, e.g. `"a chunked dataset"`.
        content: &'static str,
        /// The library-version label of the format that content requires.
        needs: &'static str,
        /// The library-version label of the format the bounds resolved to.
        writing: &'static str,
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
    /// An attribute's serialized message is larger than the version 2 object
    /// header's 2-byte message-size field can describe, so it cannot be stored
    /// as a compact (in-header) attribute. Refused rather than written, because
    /// a truncated size field would desynchronize every message after it. The
    /// fields carry the attribute name and its serialized message size in
    /// bytes; the limit is [`OBJECT_HEADER_MESSAGE_MAX`].
    ///
    /// A backstop rather than an outcome you should expect to see: the
    /// whole-file writer selects dense (fractal-heap) storage for exactly the
    /// attributes that would trip this, so no input reaches it today. It stays
    /// because the limit it describes is a real property of the object header,
    /// and any future path that must keep an attribute compact needs it.
    AttributeMessageTooLarge {
        /// The attribute's name.
        name: String,
        /// The attribute message's serialized size in bytes.
        size: usize,
    },
    /// An object-header message is larger than the version 2 object header's
    /// 2-byte message-size field can describe. This is the whole-file writer's
    /// backstop against emitting a truncated size field, covering every message
    /// it builds; callers that can name the offending object (for example an
    /// attribute) report a more specific error first. The in-place editor
    /// encodes headers separately and refuses the same condition with
    /// [`Error::EditUnsupported`](crate::Error::EditUnsupported). The fields
    /// carry the message type code and the message's serialized size in bytes;
    /// the limit is [`OBJECT_HEADER_MESSAGE_MAX`].
    ObjectHeaderMessageTooLarge {
        /// The header message's type code (see the HDF5 message type table).
        message_type: u16,
        /// The message's serialized size in bytes.
        size: usize,
    },
    /// An attribute's name, datatype or dataspace is longer than the attribute
    /// message can describe: each of the three has a 2-byte length field, so a
    /// longer one would truncate and produce a message that decodes as something
    /// else. Refused rather than written.
    ///
    /// This bounds the attribute's *description*, not its data. An attribute
    /// whose data is arbitrarily large is written to dense storage as a
    /// fractal-heap huge object.
    ///
    /// Reported by the whole-file writer (including through `repack`), which
    /// sends any attribute too large for an object-header message to dense
    /// storage and so reaches this check.
    ///
    /// In practice `field` is always `"name"`. Every attribute this crate writes
    /// is built from an [`AttrValue`](crate::AttrValue) — including the ones
    /// `repack` carries over, which it refuses outright if it cannot represent
    /// them — and no variant of it produces a datatype past 20 bytes or a
    /// dataspace past 12, whatever the data. The other two are still checked,
    /// because the message encodes all three lengths the same way, and named
    /// rather than merged so that a datatype which does one day reach the limit
    /// is diagnosable from the error rather than mistaken for a long name.
    AttributeFieldTooLong {
        /// The attribute's name.
        name: String,
        /// Which of the message's three 2-byte length fields overflowed:
        /// `"name"`, `"datatype"` or `"dataspace"`. Diagnostic detail rather than
        /// a discriminator to branch on, since only `"name"` is reachable today.
        field: &'static str,
        /// That field's encoded length in bytes.
        size: usize,
        /// The largest length the message can describe, in bytes.
        limit: usize,
    },
    /// An object's attributes need more space than a dense attribute heap can
    /// address: its offsets are 40 bits wide, so the blocks holding the
    /// attributes cannot span more than `limit` bytes between them. Reaching this
    /// takes about a terabyte of attributes on a single object.
    ///
    /// A set that fits the heap but not the host reports
    /// [`FormatError::ValueTooLargeForPlatform`] instead, so the limit named here
    /// is always the one that actually applied.
    DenseAttributeHeapTooLarge {
        /// The heap address space, in bytes.
        limit: u64,
    },
}

/// The largest message a version 2 object header can describe: its per-message
/// size field is 2 bytes wide. A message past this must be refused rather than
/// written with a truncated length (see
/// [`FormatError::ObjectHeaderMessageTooLarge`]).
pub const OBJECT_HEADER_MESSAGE_MAX: usize = u16::MAX as usize;

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
            FormatError::InvalidFileSpacePageSize(p) => {
                write!(
                    f,
                    "invalid file-space page size {p}: must be a power of two >= 512"
                )
            }
            FormatError::UserblockNotPageAligned(userblock, page_size) => {
                write!(
                    f,
                    "userblock of {userblock} bytes is not a whole number of {page_size}-byte \
                     file-space pages: a paged file measures its pages from the file base, so \
                     the userblock must be a multiple of the page size (or zero)"
                )
            }
            FormatError::InvalidUserblockSize(size) => {
                write!(
                    f,
                    "invalid userblock size {size}: must be zero or a power of two >= 512"
                )
            }
            FormatError::UserblockContentTooLarge { content, userblock } => {
                write!(
                    f,
                    "{content} bytes of userblock content do not fit a userblock of {userblock} \
                     bytes"
                )
            }
            FormatError::InvalidFreeSpaceManager => {
                write!(f, "malformed free-space manager block (FSHD/FSSE)")
            }
            FormatError::EnumBaseNotInteger => {
                write!(
                    f,
                    "an enumeration's base type must be an integer (fixed-point) type"
                )
            }
            FormatError::EnumMemberValueSize(name, expected, actual) => {
                write!(
                    f,
                    "enumeration member '{name}' has a {actual}-byte value, but its base type is \
                     {expected} bytes"
                )
            }
            FormatError::EnumMemberValueRange(name, value, size) => {
                write!(
                    f,
                    "enumeration member '{name}' value {value} does not fit in its \
                     {size}-byte base type"
                )
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
            FormatError::UnexpectedHugeObjectBTree {
                tree_type,
                record_size,
                required,
            } => {
                write!(
                    f,
                    "fractal-heap huge-objects B-tree is not the expected record type 1: \
                     type {tree_type}, records of {record_size} bytes (type 1 needs {required})"
                )
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
                     cannot be satisfied: this crate writes the v1.8 and {writes} formats"
                )
            }
            FormatError::LibverTooOldForContent {
                content,
                needs,
                writing,
            } => {
                write!(
                    f,
                    "{content} requires the {needs} format, but the requested \
                     library-version bounds write {writing}"
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
            FormatError::AttributeMessageTooLarge { name, size } => {
                write!(
                    f,
                    "attribute {name:?} serializes to {size} bytes, past the \
                     {OBJECT_HEADER_MESSAGE_MAX}-byte limit of the object header's \
                     message size field"
                )
            }
            FormatError::ObjectHeaderMessageTooLarge { message_type, size } => {
                write!(
                    f,
                    "object header message {message_type:#06x} is {size} bytes, past the \
                     {OBJECT_HEADER_MESSAGE_MAX}-byte limit of the object header's \
                     message size field"
                )
            }
            FormatError::AttributeFieldTooLong {
                name,
                field,
                size,
                limit,
            } => {
                write!(
                    f,
                    "attribute {name:?} has a {size}-byte {field}, past the {limit}-byte limit of \
                     the attribute message's {field} size field"
                )
            }
            FormatError::DenseAttributeHeapTooLarge { limit } => {
                write!(
                    f,
                    "these attributes need more than the {limit}-byte address space of a dense \
                     attribute heap"
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
    /// The file or dataset is not a supported target for the SWMR append writer
    /// (e.g. a userblock or non-latest-format file, or a dataset that is
    /// filtered, not rank-1 with an unlimited dimension, or not
    /// Extensible-Array indexed). The payload is a human-readable reason.
    SwmrAppendUnsupported(&'static str),
    /// The dataset is not a supported target for
    /// [`Dataset::append_staged`](crate::Dataset::append_staged) — for
    /// example a dataset that is not chunked, not extensible along its first
    /// dimension, not indexed by an Extensible Array, higher than rank 1, uses a
    /// filter this engine cannot re-encode, has a big-endian on-disk element
    /// datatype (for a raw append), or has more than one hard link. The payload
    /// is a human-readable reason.
    AppendUnsupported(&'static str),
    /// The dataset or file is not a supported target for the fast, immediate
    /// in-place append
    /// ([`Dataset::append`](crate::Dataset::append)) — for
    /// example a userblock or non-latest-format file, a dataset whose
    /// Extensible-Array index is not yet allocated, one that is not rank-1 /
    /// unlimited / Extensible-Array indexed, one reachable through more than one
    /// hard link, or a path an uncommitted staged edit in the same session will
    /// relocate or delete. Distinct from [`AppendUnsupported`](Self::AppendUnsupported)
    /// so a caller can catch this fast-path refusal and fall back to the staged
    /// [`Dataset::append_staged`](crate::Dataset::append_staged). The
    /// payload is a human-readable reason.
    AppendInPlaceUnsupported(&'static str),
    /// The file or the requested object is not a supported target for the
    /// in-place editor ([`crate::File::open_rw`]) — for example a userblock or
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
    /// OS advisory lock — for a writer ([`crate::File::open_swmr_writer`],
    /// [`crate::File::open_rw`]) this means another writer or reader is active;
    /// for a plain reader it means a writer is active. The lock is released
    /// automatically when the holder's process exits, so a crashed writer does
    /// not leave a stale lock. Locking can be disabled per open with
    /// [`crate::FileLocking::Disabled`] or globally with
    /// `HDF5_USE_FILE_LOCKING=FALSE`. The payload is a human-readable reason.
    FileLocked(String),
    /// The file could not be opened because its superblock's status-flags byte
    /// marks it as held by a writer — the durable flag
    /// [`crate::File::open_swmr_writer`] raises, and the reference C library
    /// raises for any writer. Unlike [`FileLocked`](Self::FileLocked) this
    /// outlives the process that set it, so it means either that a writer is
    /// active *or* that one exited without clearing it; the payload names
    /// [`crate::File::clear_swmr_flag`] (the `h5clear -s` equivalent) as the
    /// recovery for the latter. A live SWMR writer can still be followed with
    /// [`crate::File::open_swmr`]. The payload is a human-readable reason.
    FileMarkedInUse(String),
}

#[cfg(feature = "std")]
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Format(e) => write!(f, "HDF5 format error: {e}"),
            Error::NotADataset(path) => write!(f, "not a dataset: {path}"),
            Error::MissingMessage(mt) => write!(f, "missing required message: {mt}"),
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
            Error::FileMarkedInUse(reason) => write!(f, "file is marked in use: {reason}"),
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
