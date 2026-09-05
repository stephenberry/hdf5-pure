//! A rectangular region of a dataset — `start[i] .. start[i] + count[i]` along
//! every dimension — and where it lies in dense storage.
//!
//! A row window is the region that spans every inner dimension. The chunked
//! reader meets a region chunk by chunk (`chunked_read`); a compact or
//! contiguous layout stores the dataset as one row-major array, and a region of
//! it is the runs [`runs`] yields.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec, vec::Vec};

use crate::convert::TryToUsize;
use crate::error::FormatError;

/// Check that `start` and `count` have one entry per axis of `dims` and that
/// every axis ends inside its dimension, or say which does not, worded for the
/// caller who asked for the region.
///
/// A region is not clamped the way a row window is. A row keeps its full inner
/// shape however the window is cut, so a clamped window is still rows; a
/// clamped region would come back in a shape the caller did not ask for, so it
/// is refused instead. A zero `count` is fine anywhere, including at the edge.
pub(crate) fn fits(dims: &[u64], start: &[u64], count: &[u64]) -> Result<(), String> {
    if start.len() != dims.len() || count.len() != dims.len() {
        return Err(format!(
            "start has {} entries and count {} for a dataset of rank {}",
            start.len(),
            count.len(),
            dims.len()
        ));
    }
    for (axis, ((&dim, &from), &len)) in dims.iter().zip(start).zip(count).enumerate() {
        match from.checked_add(len) {
            Some(end) if end <= dim => {}
            _ => {
                return Err(format!(
                    "axis {axis}: {from}..{} runs past its extent of {dim}",
                    from.saturating_add(len)
                ));
            }
        }
    }
    Ok(())
}

/// One contiguous stretch of a region in a row-major array: `len` elements,
/// from element `src` of the array to element `dst` of the region's own buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Run {
    pub src: u64,
    pub dst: usize,
    pub len: usize,
}

/// The runs of the region `start .. start + count` over a row-major array of
/// `dims`, in storage order.
///
/// Trailing axes the region spans fully are contiguous in storage, so a run is
/// cut at the last axis the region does not span — `count` elements there times
/// every dimension after it — and there is one run per coordinate of the axes
/// before. A region spanning everything is one run; an empty one has none.
///
/// The caller has checked the region against `dims` ([`fits`]); what is checked
/// here is arithmetic. The array's strides are kept in `u64`, since a dataset
/// may hold more elements than this platform addresses, and a crafted dataspace
/// whose extent overflows errors rather than wrapping. The region's own buffer
/// is allocated, so its strides and offsets are `usize`.
pub(crate) fn runs(dims: &[u64], start: &[u64], count: &[u64]) -> Result<Runs, FormatError> {
    let rank = dims.len();
    debug_assert!(
        start.len() == rank && count.len() == rank,
        "`fits` or the row-window wrapper guarantees matching lengths"
    );

    let mut out_elems = 1usize;
    for &c in count {
        out_elems = out_elems
            .checked_mul(c.to_usize()?)
            .ok_or(FormatError::OffsetOverflow {
                offset: out_elems as u64,
                length: c,
            })?;
    }
    if out_elems == 0 {
        return Ok(Runs::none());
    }
    if rank == 0 {
        // A scalar: the one element, whole.
        return Ok(Runs::one(1));
    }

    // Strides in elements; the array's are checked up to and including its
    // whole extent, so every offset inside it fits.
    let mut array_strides = vec![1u64; rank];
    for i in (0..rank - 1).rev() {
        array_strides[i] =
            array_strides[i + 1]
                .checked_mul(dims[i + 1])
                .ok_or(FormatError::OffsetOverflow {
                    offset: array_strides[i + 1],
                    length: dims[i + 1],
                })?;
    }
    array_strides[0]
        .checked_mul(dims[0])
        .ok_or(FormatError::OffsetOverflow {
            offset: array_strides[0],
            length: dims[0],
        })?;
    let mut out_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        // Each stride is a product of `count` entries, bounded by `out_elems`.
        out_strides[i] = out_strides[i + 1] * count[i + 1].to_usize()?;
    }

    // The last axis the region does not span; everything after it is contiguous
    // in both the array and the region. A region spanning everything pivots on
    // axis 0 and is one run.
    let pivot = (0..rank)
        .rev()
        .find(|&i| start[i] != 0 || count[i] != dims[i])
        .unwrap_or(0);
    let len = count[pivot].to_usize()? * out_strides[pivot];
    // Unreachable once the extent check above passed — every offset inside the
    // array fits — but the fold stays checked rather than trusting that.
    let base = start
        .iter()
        .zip(&array_strides)
        .try_fold(0u64, |acc, (&s, &stride)| {
            acc.checked_add(s.checked_mul(stride)?)
        })
        .ok_or(FormatError::OffsetOverflow {
            offset: start[pivot],
            length: array_strides[pivot],
        })?;
    // One run per coordinate of the axes before the pivot — a factor of
    // `out_elems`, so the product fits.
    let remaining = count[..pivot]
        .iter()
        .try_fold(1usize, |acc, &c| c.to_usize().map(|c| acc * c))?;

    Ok(Runs {
        coord: vec![0; pivot],
        count: count[..pivot].to_vec(),
        array_strides: array_strides[..pivot].to_vec(),
        out_strides: out_strides[..pivot].to_vec(),
        base,
        len,
        remaining,
    })
}

/// The runs of a region, in storage order; see [`runs`].
#[derive(Debug)]
pub(crate) struct Runs {
    /// Odometer over the axes before the pivot, in region coordinates.
    coord: Vec<usize>,
    count: Vec<u64>,
    array_strides: Vec<u64>,
    out_strides: Vec<usize>,
    /// Element offset of the first run in the array.
    base: u64,
    /// Elements per run.
    len: usize,
    remaining: usize,
}

impl Runs {
    fn none() -> Self {
        Self::one(0).with_remaining(0)
    }

    fn one(len: usize) -> Self {
        Runs {
            coord: Vec::new(),
            count: Vec::new(),
            array_strides: Vec::new(),
            out_strides: Vec::new(),
            base: 0,
            len,
            remaining: 1,
        }
    }

    fn with_remaining(mut self, remaining: usize) -> Self {
        self.remaining = remaining;
        self
    }
}

impl Iterator for Runs {
    type Item = Run;

    fn next(&mut self) -> Option<Run> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        // Offsets inside extents already checked in `runs`, so plain arithmetic.
        let mut src = self.base;
        let mut dst = 0usize;
        for (i, &k) in self.coord.iter().enumerate() {
            src += k as u64 * self.array_strides[i];
            dst += k * self.out_strides[i];
        }
        // Advance the odometer; the last axis before the pivot varies fastest.
        for i in (0..self.coord.len()).rev() {
            self.coord[i] += 1;
            if (self.coord[i] as u64) < self.count[i] {
                break;
            }
            self.coord[i] = 0;
        }
        Some(Run {
            src,
            dst,
            len: self.len,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(dims: &[u64], start: &[u64], count: &[u64]) -> Vec<(u64, usize, usize)> {
        runs(dims, start, count)
            .unwrap()
            .map(|r| (r.src, r.dst, r.len))
            .collect()
    }

    #[test]
    fn a_region_that_ends_inside_every_axis_fits() {
        assert_eq!(fits(&[4, 6], &[1, 2], &[3, 4]), Ok(()));
        // Zero elements at the very edge is still inside.
        assert_eq!(fits(&[4, 6], &[4, 6], &[0, 0]), Ok(()));
        // A scalar has no axes to name.
        assert_eq!(fits(&[], &[], &[]), Ok(()));
    }

    #[test]
    fn a_region_past_an_edge_or_of_another_rank_is_named() {
        let past = fits(&[4, 6], &[1, 5], &[3, 2]).unwrap_err();
        assert!(
            past.contains("axis 1") && past.contains("5..7") && past.contains("6"),
            "{past}"
        );
        let rank = fits(&[4, 6], &[1], &[3, 2]).unwrap_err();
        assert!(rank.contains("rank 2"), "{rank}");
        // An overflowing end is past the edge, not wrapped back inside it.
        assert!(fits(&[4], &[u64::MAX], &[2]).is_err());
    }

    #[test]
    fn a_partial_inner_axis_cuts_one_run_per_outer_coordinate() {
        // [4, 6] array, box of 2 rows by 3 columns at (1, 2): the inner axis is
        // not spanned, so each row of the box is its own run.
        assert_eq!(
            collect(&[4, 6], &[1, 2], &[2, 3]),
            vec![(8, 0, 3), (14, 3, 3)]
        );
        // An axis cut from its start is cut all the same.
        assert_eq!(
            collect(&[4, 6], &[0, 0], &[2, 3]),
            vec![(0, 0, 3), (6, 3, 3)]
        );
    }

    #[test]
    fn spanned_trailing_axes_fold_into_the_run() {
        // Whole rows: one run per band, however many columns.
        assert_eq!(collect(&[4, 6], &[1, 0], &[2, 6]), vec![(6, 0, 12)]);
        // [2, 3, 4] with the last axis spanned: the pivot is axis 1, a run is
        // `count[1] * dims[2]` elements, one per coordinate of axis 0.
        assert_eq!(
            collect(&[2, 3, 4], &[0, 1, 0], &[2, 2, 4]),
            vec![(4, 0, 8), (16, 8, 8)]
        );
        // Everything: one run of the whole array.
        assert_eq!(collect(&[4, 6], &[0, 0], &[4, 6]), vec![(0, 0, 24)]);
    }

    #[test]
    fn an_empty_region_has_no_runs_and_a_scalar_has_one() {
        assert!(collect(&[4, 6], &[1, 2], &[0, 3]).is_empty());
        assert_eq!(collect(&[], &[], &[]), vec![(0, 0, 1)]);
    }

    #[test]
    fn an_extent_that_overflows_errors_instead_of_wrapping() {
        let err = runs(&[u64::MAX, u64::MAX], &[0, 0], &[1, 1]).unwrap_err();
        assert!(matches!(err, FormatError::OffsetOverflow { .. }), "{err:?}");
    }
}
