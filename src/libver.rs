//! Library-version bounds — the [`H5F_libver_t`] concept.
//!
//! Every HDF5 file is encoded with a set of *format versions* (superblock,
//! object headers, message encodings). A reader needs a library new enough to
//! understand those versions, and the HDF5 C API lets a writer bound which
//! versions a new file may use via `H5Pset_libver_bounds`. [`LibVer`] names the
//! release boundaries at which the on-disk format changed, so callers of this
//! crate can ask which format an existing file requires
//! ([`crate::File::libver_bound`]) or constrain what a new file may emit
//! ([`crate::FileBuilder::with_libver_bounds`]).
//!
//! [`H5F_libver_t`]: https://portal.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html

/// A library-version boundary, mirroring HDF5's `H5F_libver_t`.
///
/// Variants are ordered oldest to newest; a later variant understands strictly
/// more of the format than an earlier one. `LibVer` derives `Ord` on that
/// ordering, so bounds can be compared directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LibVer {
    /// The earliest format (HDF5 1.0+): version 0/1 superblock, v1
    /// symbol-table groups. Readable by every released HDF5 library.
    Earliest,
    /// HDF5 1.8: version 2 superblock and the "new style" (version 2) object
    /// headers, dense link/attribute storage, and the v2 B-tree indices.
    V18,
    /// HDF5 1.10: version 3 superblock, plus SWMR and the extensible/fixed
    /// array chunk indices. This is the format this crate's writer emits.
    V110,
    /// HDF5 1.12.
    V112,
    /// HDF5 1.14.
    V114,
}

impl LibVer {
    /// The newest boundary this enum knows about — the meaning of
    /// `H5F_LIBVER_LATEST`. Tracks the highest concrete variant.
    pub const LATEST: LibVer = LibVer::V114;

    /// The on-disk format this crate's [`FileBuilder`](crate::FileBuilder)
    /// produces: the version 3 superblock introduced in HDF5 1.10.
    pub const WRITER_OUTPUT: LibVer = LibVer::V110;

    /// The minimum library version required to read a file with the given
    /// superblock version — i.e. the *low bound* the on-disk format implies.
    ///
    /// Superblock 0/1 → [`Earliest`](LibVer::Earliest); 2 → [`V18`](LibVer::V18);
    /// 3 and anything newer → [`V110`](LibVer::V110).
    pub fn from_superblock_version(version: u8) -> LibVer {
        match version {
            0 | 1 => LibVer::Earliest,
            2 => LibVer::V18,
            _ => LibVer::V110,
        }
    }

    /// A short, stable label for diagnostics (e.g. error messages).
    pub fn name(self) -> &'static str {
        match self {
            LibVer::Earliest => "earliest",
            LibVer::V18 => "v1.8",
            LibVer::V110 => "v1.10",
            LibVer::V112 => "v1.12",
            LibVer::V114 => "v1.14",
        }
    }
}
