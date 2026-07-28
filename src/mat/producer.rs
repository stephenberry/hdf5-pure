//! Staging a dataset whose bytes are produced at write time.
//!
//! A [`DataProducer`] hands the writer one block of raw elements at a time,
//! during the emission pass, so a dataset written through
//! [`MatBuilder::write_blocks`](crate::mat::MatBuilder::write_blocks) is never
//! fully resident. Paired with
//! [`MatBuilder::finish_to`](crate::mat::MatBuilder::finish_to) — which does not
//! hold the assembled file either — a `.mat` of any size can be written in
//! roughly one block of memory.
//!
//! The dataset that comes out is byte-for-byte the one the ordinary writers
//! produce for the same content: contiguous storage, laid out from the shape
//! alone, with the blocks written back to back into the region the layout
//! reserved. The choice of API changes the peak memory and nothing else.
//!
//! # Uncompressed only
//!
//! The writer computes every object's address before it emits a byte, which
//! means it needs the data region's exact size up front. Uncompressed, that is
//! pure geometry. Compressed, it is not knowable without compressing — which
//! would buffer the very data this path exists to avoid. So a producer-backed
//! dataset is always stored unfiltered, and asking for one on a builder
//! configured for deflate is refused rather than silently downgraded.
//!
//! # Element order
//!
//! MATLAB is column-major and HDF5 is row-major, so a MATLAB array is stored
//! with its dimensions reversed and its elements in the very order MATLAB reads
//! them. A producer therefore emits elements in **MATLAB's linear order**: the
//! first index varies fastest. Block `i` is a contiguous run of that order,
//! continuing where block `i - 1` stopped.
//!
//! For an acquisition that matters, because it fixes which shape to ask for.
//! `[channels, samples]` puts all of a timestep's channels next to each other,
//! so the blocks run forward through time — the order the samples arrive in. The
//! transpose, `[samples, channels]`, stores channel 0's entire history before
//! channel 1's, so no producer can emit it as an acquisition proceeds.

use crate::chunked_write::ChunkProvider;
use crate::datatype::Datatype;
use crate::error::FormatError;
use crate::mat::class::MatClass;
use crate::mat::error::MatError;
use crate::type_builders::{
    CompoundTypeBuilder, make_f32_type, make_f64_type, make_i8_type, make_i16_type, make_i32_type,
    make_i64_type, make_u8_type, make_u16_type, make_u32_type, make_u64_type,
};
use std::sync::{Arc, Mutex};

/// Byte size a block aims for. Large enough that a producer call's overhead is
/// negligible, small enough to be an unremarkable allocation. Blocks are whole
/// numbers of elements, so an element wider than this gets one element per block
/// rather than a split that would cut an element in half.
const TARGET_BLOCK_BYTES: u64 = 1 << 20;

/// Yields a dataset's raw, uncompressed element bytes on demand, one block at a
/// time.
///
/// The writer calls [`block_bytes`](Self::block_bytes) once per block, in
/// ascending index order, during the emission pass — never during layout, which
/// works from the dataset's shape alone. Implementations therefore do not need
/// to be re-runnable, and generating each block on the fly is the intended use.
///
/// `Send + Sync` is required because a staged producer is owned by the builder,
/// and the builder would otherwise lose those auto-traits.
pub trait DataProducer: Send + Sync {
    /// Append block `index`'s raw little-endian element bytes to `out`, in
    /// MATLAB's linear element order (see the [module docs](self)).
    ///
    /// `out` is handed over empty and is the same buffer on every call, so
    /// appending costs one allocation for the whole dataset. Write exactly
    /// [`Blocking::block_len`] bytes for this index; any other count is refused
    /// with [`MatError::BlockSizeMismatch`] rather than written, because a block
    /// of the wrong size shifts every address after it.
    fn block_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), MatError>;
}

/// How a producer-backed dataset was split for the write pass.
///
/// Returned by [`MatBuilder::write_blocks`](crate::mat::MatBuilder::write_blocks)
/// and computable in advance with [`Blocking::plan`], so a producer can be built
/// against the same blocking the writer will use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Blocking {
    /// Number of blocks the producer will be asked for: indices `0..block_count`.
    /// Zero for an empty dataset, which is written as a MATLAB empty marker and
    /// never calls the producer.
    pub block_count: usize,
    /// Elements in every block but the last.
    pub block_elements: u64,
    /// Elements in the last block. Equal to `block_elements` when the dataset
    /// divides evenly.
    pub last_block_elements: u64,
    /// Bytes one element occupies. A complex element counts both components.
    pub element_size: usize,
}

impl Blocking {
    /// Bytes the producer must write for block `index`.
    ///
    /// Every block but the last is the same size; the last is whatever remains.
    /// Out-of-range indices report zero, since the writer never asks for them.
    pub fn block_len(&self, index: usize) -> usize {
        let elements = if index + 1 == self.block_count {
            self.last_block_elements
        } else if index < self.block_count {
            self.block_elements
        } else {
            0
        };
        // A planned block fits `usize` — that is checked when the blocking is
        // built. The fields are public, though, so a caller can hand back a
        // modified copy; reporting `usize::MAX` for one no host could hold means
        // no producer can satisfy it, and the write fails as a size mismatch
        // rather than on a number that quietly wrapped.
        usize::try_from(elements)
            .ok()
            .and_then(|n| n.checked_mul(self.element_size))
            .unwrap_or(usize::MAX)
    }

    /// Total bytes the dataset's elements occupy.
    pub fn total_len(&self) -> u64 {
        if self.block_count == 0 {
            return 0;
        }
        (self.block_elements * (self.block_count as u64 - 1) + self.last_block_elements)
            * self.element_size as u64
    }

    /// The blocking [`write_blocks`](crate::mat::MatBuilder::write_blocks) will
    /// choose for a dataset of this MATLAB shape and element type.
    ///
    /// Deterministic, and computed from the shape alone, so a producer can be
    /// built against it before the dataset is staged.
    pub fn plan<T: BlockElement>(matlab_dims: &[usize]) -> Result<Blocking, MatError> {
        plan_blocking(total_elements(matlab_dims), T::ELEMENT_SIZE)
    }
}

/// Elements a MATLAB shape holds. An empty shape (`[]`) is one element, matching
/// [`matrix_dims`](crate::mat::dims::matrix_dims)'s scalar collapse.
///
/// Saturates rather than overflows: dimensions come from the caller, so an
/// absurd shape must reach the planner's refusal rather than panic a debug build
/// on the way there. A saturated count is larger than any host can address, so it
/// is refused for the right reason.
pub(crate) fn total_elements(matlab_dims: &[usize]) -> u64 {
    matlab_dims
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))
        .unwrap_or(u64::MAX)
}

/// Split `total` elements into blocks of about [`TARGET_BLOCK_BYTES`].
///
/// Blocks are whole numbers of elements and successive runs of the dataset's
/// linear order, which is what lets a producer treat block `i` as "the next
/// chunk of my stream" rather than having to index into a grid.
pub(crate) fn plan_blocking(total: u64, element_size: usize) -> Result<Blocking, MatError> {
    if total == 0 || element_size == 0 {
        return Ok(Blocking {
            block_count: 0,
            block_elements: 0,
            last_block_elements: 0,
            element_size,
        });
    }
    // The data region's byte count is the first thing that has to be real: it
    // goes into the object header as a `u64`, and every later size here is
    // derived from it. A shape whose elements overflow that is refused before any
    // of the arithmetic below can wrap.
    let total_bytes = total
        .checked_mul(element_size as u64)
        .ok_or_else(|| too_large(u64::MAX))?;

    let per_block = (TARGET_BLOCK_BYTES / element_size as u64).clamp(1, total);
    let block_count = total.div_ceil(per_block);
    let last = total - (block_count - 1) * per_block;

    // The emitter builds one block in memory at a time, so the block — not the
    // dataset — has to fit this host. Only reachable on a 32-bit target, and only
    // for an element wider than the target block size.
    let block_bytes = per_block * element_size as u64;
    if usize::try_from(block_bytes).is_err() {
        return Err(too_large(block_bytes));
    }
    debug_assert!(total_bytes >= block_bytes);

    Ok(Blocking {
        block_count: usize::try_from(block_count).map_err(|_| too_large(block_count))?,
        block_elements: per_block,
        last_block_elements: last,
        element_size,
    })
}

fn too_large(value: u64) -> MatError {
    MatError::Hdf5(crate::error::Error::Format(
        FormatError::ValueTooLargeForPlatform {
            value,
            target: "usize",
        },
    ))
}

/// Adapts a [`DataProducer`] to the writer's block provider: checks each block's
/// length and carries a producer's own error back out.
///
/// The error detour exists because the writer's provider seam speaks
/// [`FormatError`], which cannot carry a [`MatError`]. Stashing the real error
/// here and swapping it back in `MatBuilder`'s finalizers keeps a producer's
/// failure intact instead of flattening it to a message.
pub(crate) struct ProducerChunks {
    pub(crate) producer: Box<dyn DataProducer>,
    pub(crate) blocking: Blocking,
    pub(crate) error: Arc<Mutex<Option<MatError>>>,
}

impl ProducerChunks {
    /// Record `error` (the first one wins) and return the placeholder the writer
    /// will carry until the finalizer swaps the real one back in.
    fn fail(&self, error: MatError) -> FormatError {
        if let Ok(mut slot) = self.error.lock() {
            slot.get_or_insert(error);
        }
        FormatError::SerializationError("a dataset's block producer failed".into())
    }
}

impl ChunkProvider for ProducerChunks {
    fn chunk_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), FormatError> {
        let expected = self.blocking.block_len(index);
        if let Err(e) = self.producer.block_bytes(index, out) {
            return Err(self.fail(e));
        }
        if out.len() != expected {
            return Err(self.fail(MatError::BlockSizeMismatch {
                block: index,
                expected,
                actual: out.len(),
            }));
        }
        Ok(())
    }
}

mod sealed {
    pub trait Sealed {}
}

/// An element type a producer-backed dataset can hold.
///
/// Implemented for the ten numeric widths MATLAB has classes for, for `bool`
/// (MATLAB `logical`), and for the `(T, T)` pair of any of those widths, which
/// is a complex array of that class — the same `(real, imag)` framing
/// [`write_complex_f64`](crate::mat::MatBuilder::write_complex_f64) and its
/// siblings take. Sealed: the set of MATLAB classes is fixed by the format.
pub trait BlockElement: sealed::Sealed {
    /// The MATLAB class the dataset reports. For a complex pair this is the
    /// *component* class, which is how MATLAB tells `complex(int16(..))` from a
    /// complex `double`.
    const CLASS: MatClass;
    /// Bytes one element occupies on disk. A complex pair counts both components.
    const ELEMENT_SIZE: usize;
    /// The `MATLAB_int_decode` attribute value, for the classes that carry one.
    const INT_DECODE: Option<i32> = None;
    /// The HDF5 datatype of one element.
    fn datatype() -> Datatype;
}

macro_rules! block_elements {
    ($($ty:ty => $class:ident, $make:ident),* $(,)?) => {
        $(
            impl sealed::Sealed for $ty {}
            impl BlockElement for $ty {
                const CLASS: MatClass = MatClass::$class;
                const ELEMENT_SIZE: usize = size_of::<$ty>();
                fn datatype() -> Datatype {
                    $make()
                }
            }

            impl sealed::Sealed for ($ty, $ty) {}
            impl BlockElement for ($ty, $ty) {
                const CLASS: MatClass = MatClass::$class;
                const ELEMENT_SIZE: usize = 2 * size_of::<$ty>();
                fn datatype() -> Datatype {
                    CompoundTypeBuilder::new()
                        .field("real", $make())
                        .field("imag", $make())
                        .build()
                }
            }
        )*
    };
}

block_elements! {
    f64 => Double, make_f64_type,
    f32 => Single, make_f32_type,
    i8  => Int8,   make_i8_type,
    i16 => Int16,  make_i16_type,
    i32 => Int32,  make_i32_type,
    i64 => Int64,  make_i64_type,
    u8  => UInt8,  make_u8_type,
    u16 => UInt16, make_u16_type,
    u32 => UInt32, make_u32_type,
    u64 => UInt64, make_u64_type,
}

impl sealed::Sealed for bool {}
impl BlockElement for bool {
    const CLASS: MatClass = MatClass::Logical;
    const ELEMENT_SIZE: usize = 1;
    const INT_DECODE: Option<i32> = Some(1);
    fn datatype() -> Datatype {
        make_u8_type()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blocking is derived from the element count alone, so these are the
    /// numbers a caller can plan against before staging anything. Derived here
    /// by a different route than the planner uses — summing the block lengths
    /// rather than re-running its arithmetic.
    #[test]
    fn the_blocks_partition_the_dataset() {
        // 3_000_000 f64 = 24 MB; a 1 MiB block is 131_072 elements.
        let b = plan_blocking(3_000_000, 8).unwrap();
        assert_eq!(b.block_elements, 131_072);
        assert!(b.block_count > 1, "the fixture must span several blocks");

        let total: usize = (0..b.block_count).map(|i| b.block_len(i)).sum();
        assert_eq!(total as u64, 3_000_000 * 8);
        assert_eq!(total as u64, b.total_len());
        // Every block but the last is full, and the last is a partial remainder
        // rather than a padded one.
        for i in 0..b.block_count - 1 {
            assert_eq!(b.block_len(i), 131_072 * 8);
        }
        assert!(b.last_block_elements < b.block_elements);
        assert_eq!(b.block_len(b.block_count), 0, "no block past the last");
    }

    /// An element wider than the target block is not split: a block is a whole
    /// number of elements, so one element is the floor.
    #[test]
    fn an_element_wider_than_the_target_becomes_the_block() {
        let huge = (TARGET_BLOCK_BYTES + 8) as usize;
        let b = plan_blocking(3, huge).unwrap();
        assert_eq!(b.block_elements, 1);
        assert_eq!(b.block_count, 3);
        assert_eq!(b.block_len(0), huge);
    }

    /// A dataset that fits one block has no short tail, and `last` still reports
    /// the real count rather than zero.
    #[test]
    fn a_dataset_smaller_than_a_block_is_one_full_block() {
        let b = plan_blocking(8, 8).unwrap();
        assert_eq!(b.block_count, 1);
        assert_eq!(b.block_elements, 8);
        assert_eq!(b.last_block_elements, 8);
        assert_eq!(b.block_len(0), 64);
        assert_eq!(b.total_len(), 64);
    }

    /// An empty dataset asks the producer for nothing at all.
    #[test]
    fn an_empty_shape_plans_no_blocks() {
        let b = plan_blocking(total_elements(&[0, 0]), 8).unwrap();
        assert_eq!(b.block_count, 0);
        assert_eq!(b.block_len(0), 0);
        assert_eq!(b.total_len(), 0);
    }

    /// A shape nothing could hold is refused rather than wrapped into a small
    /// plausible one. `total_len` and the block arithmetic are all derived from
    /// the byte count, so it has to be the thing that is checked.
    #[test]
    fn a_shape_whose_bytes_overflow_is_refused() {
        assert!(plan_blocking(u64::MAX, 8).is_err());
        assert!(plan_blocking(u64::MAX / 8 + 1, 8).is_err());
        // One element short of the overflow still plans, and reports a total
        // that did not wrap.
        let b = plan_blocking(u64::MAX / 8, 8).unwrap();
        assert_eq!(b.total_len(), (u64::MAX / 8) * 8);

        // Dimensions are multiplied the same way, so an absurd shape reaches the
        // refusal rather than panicking a debug build on the way.
        assert_eq!(total_elements(&[usize::MAX, usize::MAX]), u64::MAX);
        assert!(Blocking::plan::<f64>(&[usize::MAX, usize::MAX]).is_err());
    }

    /// A complex pair is one element of two components, so it blocks exactly as
    /// a real array of twice the width would, and reports the component class.
    #[test]
    fn a_complex_pair_counts_both_components() {
        assert_eq!(<(i16, i16) as BlockElement>::ELEMENT_SIZE, 4);
        assert_eq!(<(i16, i16) as BlockElement>::CLASS, MatClass::Int16);
        let b = Blocking::plan::<(i16, i16)>(&[2, 10]).unwrap();
        assert_eq!(b.element_size, 4);
        assert_eq!(b.total_len(), 2 * 10 * 4);
    }

    /// `logical` is the one class that carries a decode flag, and it is stored
    /// one byte per element.
    #[test]
    fn logical_is_a_byte_with_a_decode_flag() {
        assert_eq!(<bool as BlockElement>::ELEMENT_SIZE, 1);
        assert_eq!(<bool as BlockElement>::INT_DECODE, Some(1));
        assert_eq!(<f64 as BlockElement>::INT_DECODE, None);
    }
}
