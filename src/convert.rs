//! Checked conversions from file-derived 64-bit values to platform integers.
//!
//! HDF5 stores offsets, lengths, sizes, and element counts as 64-bit values
//! (the on-disk "size of offsets" / "size of lengths" are commonly 8 bytes).
//! The reader, however, indexes an in-memory byte buffer with `usize`. On a
//! 64-bit host these conversions are infallible, but on a 32-bit (or WASM32)
//! host `usize` is 32 bits, so a `u64 as usize` cast *silently truncates* any
//! value above 4 GiB, reading from the wrong location or sizing an allocation
//! incorrectly.
//!
//! The helpers here replace those casts with fallible conversions that return
//! [`FormatError`] instead of truncating. They all return [`FormatError`], so
//! they compose with `?` in both the format layer (which returns
//! `Result<_, FormatError>`) and the high-level layer (whose `Error` has a
//! `From<FormatError>` impl).
//!
//! ## When to use these
//!
//! Use [`TryToUsize::to_usize`] / [`slice_range`] / [`u32_from`] for any value
//! that is **file-derived and not structurally bounded**: the result of
//! `read_offset`/`read_length`, a `DataLayout` address or size, a chunk offset
//! or size, a heap or collection size, `num_elements`, an element count, or any
//! arithmetic on those that becomes a slice index or allocation size.
//!
//! A narrowing `as` cast is acceptable only when the source is **provably
//! bounded on every supported target** (e.g. a `u8` "size of offsets" field that
//! is 2/4/8, a version/flags byte, a `u16` message size capped at 64 KiB, a
//! match arm keyed on the on-disk field width it casts to, or a small loop
//! counter). A `u8`/`u16` widening to `usize` never narrows and needs no guard.
//!
//! When you keep such a cast, annotate it at the site with
//! `#[expect(clippy::cast_possible_truncation /* or cast_possible_wrap */,
//! reason = "…")]`, where the `reason` states the bound that makes it safe. The
//! 32-bit CI gate (the `cast-deny-32bit` job) denies both lints, so every
//! narrowing cast must be either converted through the helpers above or
//! explicitly accounted for this way; a leftover `#[expect]` whose cast was
//! later removed fails the gate too, keeping the annotations honest.

#[cfg(not(feature = "std"))]
use core::ops::Range;
#[cfg(feature = "std")]
use std::ops::Range;

use crate::error::FormatError;

/// Fallible narrowing of a file-derived integer to the platform `usize`.
///
/// On targets where the source type already fits `usize` (e.g. `u64` on a
/// 64-bit host) this collapses to an infallible widening that the optimizer
/// removes; the error arm is cold. On a narrower target it returns
/// [`FormatError::ValueTooLargeForPlatform`] rather than truncating.
pub trait TryToUsize {
    /// Narrow `self` to `usize`, or return
    /// [`FormatError::ValueTooLargeForPlatform`] if it does not fit.
    fn to_usize(self) -> Result<usize, FormatError>;
}

impl TryToUsize for u64 {
    #[inline]
    fn to_usize(self) -> Result<usize, FormatError> {
        usize::try_from(self).map_err(|_| FormatError::ValueTooLargeForPlatform {
            value: self,
            target: "usize",
        })
    }
}

impl TryToUsize for u32 {
    #[inline]
    fn to_usize(self) -> Result<usize, FormatError> {
        // `u32` fits `usize` on every target this crate supports (>= 32-bit),
        // but routing through `try_from` keeps the API uniform and stays correct
        // on a hypothetical 16-bit target.
        usize::try_from(self).map_err(|_| FormatError::ValueTooLargeForPlatform {
            value: u64::from(self),
            target: "usize",
        })
    }
}

/// Fallibly narrow a file-derived `u64` to `u32`.
///
/// Used where the in-memory representation of a (de)compressed chunk size or
/// similar quantity is a `u32` regardless of platform pointer width. Returns
/// [`FormatError::ValueTooLargeForPlatform`] (with `target: "u32"`) instead of
/// truncating.
#[inline]
pub fn u32_from(value: u64) -> Result<u32, FormatError> {
    u32::try_from(value).map_err(|_| FormatError::ValueTooLargeForPlatform {
        value,
        target: "u32",
    })
}

/// Compute `offset .. offset + len` as a `usize` range, checking both the
/// 64-bit addition and the narrowing of each bound to `usize`.
///
/// Use this anywhere a file-derived `(offset, length)` pair becomes a slice
/// index. It guards two distinct hazards a bare `offset as usize + len as usize`
/// misses: the `u64` addition wrapping ([`FormatError::OffsetOverflow`]) and
/// either operand exceeding `usize` on a 32-bit target
/// ([`FormatError::ValueTooLargeForPlatform`]). The returned `range.end` is the
/// already-checked `usize` end bound, so a subsequent bounds check against the
/// buffer length stays truthful.
#[inline]
pub fn slice_range(offset: u64, len: u64) -> Result<Range<usize>, FormatError> {
    let end = offset.checked_add(len).ok_or(FormatError::OffsetOverflow {
        offset,
        length: len,
    })?;
    Ok(offset.to_usize()?..end.to_usize()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_values_round_trip() {
        assert_eq!(0u64.to_usize().unwrap(), 0);
        assert_eq!(1234u64.to_usize().unwrap(), 1234);
        assert_eq!(42u32.to_usize().unwrap(), 42);
        assert_eq!(u32_from(1000).unwrap(), 1000);
    }

    #[test]
    fn slice_range_basic() {
        let r = slice_range(10, 20).unwrap();
        assert_eq!(r, 10..30);
    }

    #[test]
    fn slice_range_addition_overflow_is_caught() {
        let err = slice_range(u64::MAX, 1).unwrap_err();
        assert!(matches!(err, FormatError::OffsetOverflow { .. }));
    }

    // On 64-bit hosts `u64::MAX` fits the `u64` add but not `usize`... actually
    // it does fit usize on 64-bit, so this only errors on the addition. Verify
    // the platform-narrowing guard directly with a value that overflows usize
    // only where usize < 64 bits; on 64-bit it succeeds, which is correct.
    #[test]
    fn large_u64_behaviour_matches_platform_width() {
        let big: u64 = u64::from(u32::MAX) + 1; // 2^32
        match big.to_usize() {
            // 64-bit (and wider) hosts: fits.
            Ok(v) => assert_eq!(v as u64, big),
            // 32-bit hosts: must be reported, never truncated.
            Err(e) => assert!(matches!(e, FormatError::ValueTooLargeForPlatform { .. })),
        }
    }
}
