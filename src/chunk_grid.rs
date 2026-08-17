//! Where a chunk sits in a chunk index, and what sits at a given slot.
//!
//! A Fixed Array and an Extensible Array both store their elements *positionally*:
//! an element carries a chunk's address (and, when filtered, its stored size and
//! filter mask) but never its coordinates. The slot number is the coordinates.
//! Getting that numbering wrong therefore produces a file that decodes perfectly
//! and resolves its data to the wrong addresses — which is why this rule lives in
//! one place that the writer and both readers call, rather than being restated on
//! each side (issue #299).
//!
//! Two things about the numbering are easy to get wrong, and this crate had both:
//!
//! * It is taken over the dataset's **maximum** chunk grid, not its current one.
//!   A dataset of shape `[3, 3]` with `[2, 2]` chunks holds four chunks either
//!   way, but declaring `maxshape [3, 8]` widens the grid to 2x4 and moves the
//!   two bottom chunks from slots 2 and 3 to slots 4 and 5.
//! * An Extensible Array rotates its unlimited dimension to the front first
//!   (`H5VM_swizzle_coords`), so that the one dimension free to grow is the
//!   slowest-varying and a new row of chunks lands past every existing slot
//!   rather than interleaved among them. With the unlimited dimension already
//!   first the rotation is the identity, which is why one-dimensional and
//!   grow-down-the-rows datasets were unaffected.
//!
//! The grid's own dimension counts come from the maximum extent, so an unlimited
//! dimension has no count at all. That is not a problem to work around: after the
//! rotation the unbounded dimension is the outermost one, and an outermost
//! dimension's extent never enters any multiplier.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use crate::error::FormatError;

/// How a chunk index numbers its element slots.
///
/// The two indexes this crate writes differ only in whether the unlimited
/// dimension is rotated to the front, so that difference is the whole enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GridOrder {
    /// Row-major over the maximum chunk grid, dimensions in dataset order: what
    /// a Fixed Array uses, and what an Extensible Array uses when its unlimited
    /// dimension is already the first one.
    RowMajor,
    /// Row-major over the maximum chunk grid with the unlimited dimension
    /// rotated to the front: what an Extensible Array uses.
    ///
    /// Choosing this for a dataspace with no unlimited dimension is not an
    /// error and not a special case — the rotation is simply the identity, which
    /// is also how the reference library behaves when it finds no unlimited
    /// dimension to swizzle on.
    UnlimitedFirst,
}

/// The chunk grid a chunk index numbers its slots over.
///
/// Built once from a dataset's chunk dimensions and dataspace and then asked
/// either question: [`slot_of`](Self::slot_of) for the writer, which knows a
/// chunk's coordinates and needs its element position, and
/// [`offsets_at`](Self::offsets_at) for the reader, which knows an element
/// position and needs the chunk's logical offsets.
#[derive(Debug, Clone)]
pub(crate) struct ChunkGrid {
    /// Chunk extent of each dimension, in dataset order.
    chunk_dims: Vec<u64>,
    /// The dataset's *current* extent, in dataset order. The numbering comes
    /// from the maximum grid, but a reader still has to know which of its slots
    /// the dataset actually reaches.
    dims: Vec<u64>,
    /// The dimension rotated to the front, when one is. `None` covers both
    /// [`GridOrder::RowMajor`] and an unlimited dimension that is already first,
    /// because those are the same numbering.
    rotated: Option<usize>,
    /// Slot-number multiplier per dimension, in *rotated* order.
    down: Vec<u64>,
    /// Slots the grid spans, or `None` when a dimension is unlimited. Computed
    /// in the constructor so that a count too large to represent is a refusal
    /// there rather than something an accessor has to report out of band.
    slots: Option<u64>,
}

impl ChunkGrid {
    /// Build the grid for a dataset of `dims` (its current shape) with
    /// `max_dims` (its maximum shape, absent when the dataspace records none)
    /// and `chunk_dims`, numbered in `order`.
    ///
    /// `u64::MAX` in `max_dims` is the unlimited marker, matching
    /// `H5S_UNLIMITED` and the dataspace message this crate reads and writes.
    pub(crate) fn new(
        chunk_dims: &[u64],
        dims: &[u64],
        max_dims: Option<&[u64]>,
        order: GridOrder,
    ) -> Result<Self, FormatError> {
        let rank = chunk_dims.len();
        if dims.len() != rank {
            return Err(FormatError::ChunkedReadError(format!(
                "chunk rank {rank} does not match dataspace rank {}",
                dims.len()
            )));
        }
        if let Some(m) = max_dims {
            if m.len() != rank {
                return Err(FormatError::ChunkedReadError(format!(
                    "maximum-shape rank {} does not match dataspace rank {rank}",
                    m.len()
                )));
            }
        }

        // Each dimension's chunk count comes from its maximum extent, falling
        // back to the current one for a dataspace that records no maximum.
        let mut counts = Vec::with_capacity(rank);
        for d in 0..rank {
            let chunk = chunk_dims[d];
            if chunk == 0 {
                return Err(FormatError::ChunkedReadError(
                    "chunk dimensions must all be non-zero".into(),
                ));
            }
            let extent = max_dims.map_or(dims[d], |m| m[d]);
            counts.push(if extent == u64::MAX {
                None
            } else {
                Some(extent.div_ceil(chunk))
            });
        }

        // The reference library swizzles on the *first* unlimited dimension. A
        // second one is a dataspace it indexes with a version-2 B-tree instead,
        // which this crate neither writes nor reads; it falls out below as an
        // unbounded multiplier rather than needing its own test here.
        let rotated = match order {
            GridOrder::RowMajor => None,
            GridOrder::UnlimitedFirst => {
                counts.iter().position(Option::is_none).filter(|&d| d != 0)
            }
        };

        let mut down = vec![1u64; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            let next = counts[dim_at(i + 1, rotated)].ok_or_else(|| {
                FormatError::ChunkedReadError(
                    "a chunked dataset may have only one unlimited dimension, and it must be \
                     the one the index grows along"
                        .into(),
                )
            })?;
            down[i] = down[i + 1].checked_mul(next).ok_or_else(|| {
                FormatError::ChunkedReadError(
                    "maximum shape describes more chunks than can be numbered".into(),
                )
            })?;
        }

        // `None` here means unbounded; a product too large to represent is a
        // refusal, so that the two cannot be confused by a caller that treats
        // `None` as "an unlimited dimension, so do not declare a count".
        let mut slots = Some(1u64);
        for c in &counts {
            slots = match (slots, *c) {
                (Some(acc), Some(n)) => Some(acc.checked_mul(n).ok_or_else(|| {
                    FormatError::ChunkedReadError(
                        "maximum shape describes more chunks than can be numbered".into(),
                    )
                })?),
                _ => None,
            };
        }

        Ok(Self {
            chunk_dims: chunk_dims.to_vec(),
            dims: dims.to_vec(),
            rotated,
            down,
            slots,
        })
    }

    /// The number of slots the grid spans, or `None` when a dimension is
    /// unlimited and the count is therefore unbounded.
    ///
    /// This is the element count a Fixed Array declares (`max_nchunks` in the
    /// reference library), which is why it is the maximum grid's size and not
    /// the number of chunks actually stored.
    pub(crate) fn slots(&self) -> Option<u64> {
        self.slots
    }

    /// The element slot the chunk at `coords` (chunk coordinates, not element
    /// offsets) occupies.
    pub(crate) fn slot_of(&self, coords: &[u64]) -> Result<u64, FormatError> {
        if coords.len() != self.chunk_dims.len() {
            return Err(FormatError::ChunkedReadError(
                "chunk coordinates do not match the grid's rank".into(),
            ));
        }
        let mut slot: u64 = 0;
        for (i, d) in self.down.iter().enumerate() {
            slot = coords[dim_at(i, self.rotated)]
                .checked_mul(*d)
                .and_then(|term| slot.checked_add(term))
                .ok_or_else(|| {
                    FormatError::ChunkedReadError(
                        "chunk coordinates exceed the numbering the maximum shape allows".into(),
                    )
                })?;
        }
        Ok(slot)
    }

    /// One past the highest slot the dataset's *current* extent reaches, or `0`
    /// when it holds no chunks at all.
    ///
    /// A reader walks slots, not chunks, and a maximum shape wider than the
    /// shape leaves gaps between the occupied ones — so the walk cannot stop
    /// after the chunk count. It stops here instead, which also preserves the
    /// property the count bound was there for: a chunk index published ahead of
    /// the dataspace it belongs to (an interrupted SWMR append) yields a
    /// consistent prefix rather than chunks past the dataset's bounds.
    pub(crate) fn slot_limit(&self) -> Result<u64, FormatError> {
        let mut slot = 0u64;
        for (i, d) in self.down.iter().enumerate() {
            let dim = dim_at(i, self.rotated);
            let count = self.dims[dim].div_ceil(self.chunk_dims[dim]);
            if count == 0 {
                return Ok(0);
            }
            slot = (count - 1)
                .checked_mul(*d)
                .and_then(|term| slot.checked_add(term))
                .ok_or_else(|| {
                    FormatError::ChunkedReadError(
                        "the dataset's own extent exceeds the numbering the maximum shape allows"
                            .into(),
                    )
                })?;
        }
        Ok(slot + 1)
    }

    /// Whether `offsets` (as returned by [`offsets_at`](Self::offsets_at)) lies
    /// inside the dataset's current extent.
    ///
    /// Slots between the occupied ones decode to coordinates the dataset has
    /// not grown into yet. They hold no chunk, so the question only arises for a
    /// file whose index disagrees with its dataspace; answering it here keeps
    /// such a chunk out of the assembled result instead of scattering it past
    /// the dataset's bounds.
    pub(crate) fn contains(&self, offsets: &[u64]) -> bool {
        offsets.iter().zip(&self.dims).all(|(o, d)| o < d)
    }

    /// The logical offsets — the first element's coordinate in each dimension,
    /// in elements — of the chunk stored at element `slot`.
    pub(crate) fn offsets_at(&self, slot: u64) -> Result<Vec<u64>, FormatError> {
        let rank = self.chunk_dims.len();
        let mut offsets = vec![0u64; rank];
        let mut remaining = slot;
        for (i, d) in self.down.iter().enumerate() {
            let dim = dim_at(i, self.rotated);
            offsets[dim] = (remaining / d)
                .checked_mul(self.chunk_dims[dim])
                .ok_or_else(|| {
                    FormatError::ChunkedReadError(
                        "chunk index slot resolves past the addressable dataspace".into(),
                    )
                })?;
            remaining %= d;
        }
        Ok(offsets)
    }
}

/// Which dataset dimension sits at rotated position `i`.
///
/// The rotation is the reference library's `H5VM_swizzle_coords`: dimension `u`
/// moves to the front and dimensions `0..u` each shift one place later.
/// Expressed as an index map rather than a rotated copy so that neither
/// direction allocates — both run once per chunk of every read and write, and a
/// `Vec` apiece showed up as a per-chunk allocation the bounds in
/// `tests/allocation_bounds.rs` are there to catch.
fn dim_at(i: usize, rotated: Option<usize>) -> usize {
    match rotated {
        None => i,
        Some(u) if i == 0 => u,
        Some(u) if i <= u => i - 1,
        Some(_) => i,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The plain case: no maximum shape, so the grid is the dataset's own and
    /// slots run in dense row-major order. Every pre-existing chunked dataset in
    /// this crate reads and writes through this path, so it is the one that
    /// must not move.
    #[test]
    fn without_a_maximum_shape_slots_are_dense_row_major() {
        let g = ChunkGrid::new(&[2, 2], &[3, 3], None, GridOrder::RowMajor).unwrap();
        assert_eq!(g.slots(), Some(4));
        for (coords, slot) in [([0, 0], 0), ([0, 1], 1), ([1, 0], 2), ([1, 1], 3)] {
            assert_eq!(g.slot_of(&coords).unwrap(), slot);
        }
        assert_eq!(g.offsets_at(2).unwrap(), vec![2, 0]);
    }

    /// A maximum shape wider than the shape in a *trailing* dimension spreads
    /// the slots out: this is the numbering the reference library uses and the
    /// one that made `maxshape > shape` unreadable.
    #[test]
    fn a_wider_maximum_shape_spreads_the_slots() {
        let g = ChunkGrid::new(&[2, 2], &[3, 3], Some(&[3, 8]), GridOrder::RowMajor).unwrap();
        assert_eq!(g.slots(), Some(8));
        assert_eq!(g.slot_of(&[0, 0]).unwrap(), 0);
        assert_eq!(g.slot_of(&[0, 1]).unwrap(), 1);
        assert_eq!(g.slot_of(&[1, 0]).unwrap(), 4);
        assert_eq!(g.slot_of(&[1, 1]).unwrap(), 5);
        assert_eq!(g.offsets_at(4).unwrap(), vec![2, 0]);
    }

    /// Growing the *first* dimension changes no multiplier, which is why a
    /// resizable dataset that grows down its rows read and wrote correctly all
    /// along and hid the defect in every other shape.
    #[test]
    fn a_wider_leading_dimension_leaves_the_numbering_alone() {
        let dense = ChunkGrid::new(&[2, 2], &[3, 3], None, GridOrder::RowMajor).unwrap();
        let grown = ChunkGrid::new(&[2, 2], &[3, 3], Some(&[8, 3]), GridOrder::RowMajor).unwrap();
        for coords in [[0, 0], [0, 1], [1, 0], [1, 1]] {
            assert_eq!(
                dense.slot_of(&coords).unwrap(),
                grown.slot_of(&coords).unwrap()
            );
        }
        // The slot *count* still grows: a Fixed Array has to declare room for
        // every chunk the maximum shape allows, not only the ones stored.
        assert_eq!(grown.slots(), Some(8));
    }

    /// An Extensible Array rotates its unlimited dimension to the front, so with
    /// the unlimited dimension second the two off-diagonal chunks swap slots.
    #[test]
    fn an_extensible_array_rotates_its_unlimited_dimension_to_the_front() {
        let g = ChunkGrid::new(
            &[2, 2],
            &[3, 3],
            Some(&[3, u64::MAX]),
            GridOrder::UnlimitedFirst,
        )
        .unwrap();
        assert_eq!(g.slots(), None);
        assert_eq!(g.slot_of(&[0, 0]).unwrap(), 0);
        assert_eq!(g.slot_of(&[1, 0]).unwrap(), 1);
        assert_eq!(g.slot_of(&[0, 1]).unwrap(), 2);
        assert_eq!(g.slot_of(&[1, 1]).unwrap(), 3);
        assert_eq!(g.offsets_at(1).unwrap(), vec![2, 0]);
        assert_eq!(g.offsets_at(2).unwrap(), vec![0, 2]);
    }

    /// The rotation is a rotation, not a swap: with the unlimited dimension
    /// last, the two dimensions ahead of it shift right rather than one of them
    /// trading places with it. A swap would agree with a rotation at rank 2 and
    /// disagree here.
    #[test]
    fn the_rotation_shifts_the_dimensions_ahead_of_it() {
        let g = ChunkGrid::new(
            &[1, 2, 2],
            &[2, 3, 4],
            Some(&[2, 3, u64::MAX]),
            GridOrder::UnlimitedFirst,
        )
        .unwrap();
        // Rotated grid is [unlimited, 2, 2], so the multipliers are [4, 2, 1]
        // applied to coordinates [c, a, b] of a chunk at (a, b, c).
        assert_eq!(g.slot_of(&[0, 0, 0]).unwrap(), 0);
        assert_eq!(g.slot_of(&[0, 1, 0]).unwrap(), 1);
        assert_eq!(g.slot_of(&[1, 0, 0]).unwrap(), 2);
        assert_eq!(g.slot_of(&[1, 1, 0]).unwrap(), 3);
        assert_eq!(g.slot_of(&[0, 0, 1]).unwrap(), 4);
        // A swap of dimensions 0 and 2 would put (1, 0, 0) at slot 1 and
        // (0, 0, 1) at slot 2.
        assert_eq!(g.offsets_at(2).unwrap(), vec![1, 0, 0]);
        assert_eq!(g.offsets_at(4).unwrap(), vec![0, 0, 2]);
    }

    /// An unlimited dimension that is already first needs no rotation, so the
    /// two orders agree — the case that covers every one-dimensional extensible
    /// dataset this crate has ever written.
    #[test]
    fn an_unlimited_first_dimension_needs_no_rotation() {
        let g = ChunkGrid::new(
            &[2, 2],
            &[3, 3],
            Some(&[u64::MAX, 3]),
            GridOrder::UnlimitedFirst,
        )
        .unwrap();
        assert_eq!(g.slot_of(&[0, 0]).unwrap(), 0);
        assert_eq!(g.slot_of(&[0, 1]).unwrap(), 1);
        assert_eq!(g.slot_of(&[1, 0]).unwrap(), 2);
        assert_eq!(g.slot_of(&[1, 1]).unwrap(), 3);
    }

    /// Round-tripping every slot of a grid is the property the two directions
    /// have to share; asserting it here is what keeps a later edit to one of
    /// them from silently disagreeing with the other.
    #[test]
    fn every_slot_round_trips_through_both_directions() {
        for order in [GridOrder::RowMajor, GridOrder::UnlimitedFirst] {
            for max in [
                Some(vec![4u64, 6, 8]),
                Some(vec![2, 3, 4]),
                Some(vec![u64::MAX, 6, 8]),
                Some(vec![2, u64::MAX, 8]),
                Some(vec![2, 6, u64::MAX]),
                None,
            ] {
                let Ok(g) = ChunkGrid::new(&[1, 2, 2], &[2, 3, 4], max.as_deref(), order) else {
                    // A second unbounded dimension in row-major order has no
                    // numbering; that refusal is its own test below.
                    continue;
                };
                for slot in 0..24u64 {
                    let offsets = g.offsets_at(slot).unwrap();
                    let coords: Vec<u64> = offsets
                        .iter()
                        .zip([1u64, 2, 2])
                        .map(|(o, c)| o / c)
                        .collect();
                    assert_eq!(g.slot_of(&coords).unwrap(), slot, "{order:?} {max:?}");
                }
            }
        }
    }

    /// Two unlimited dimensions have no positional numbering at all — the
    /// reference library indexes that dataspace with a version-2 B-tree, which
    /// this crate does not write. It has to be a refusal rather than a grid that
    /// silently drops one of them.
    #[test]
    fn two_unlimited_dimensions_have_no_numbering() {
        let err = ChunkGrid::new(
            &[2, 2],
            &[3, 3],
            Some(&[u64::MAX, u64::MAX]),
            GridOrder::UnlimitedFirst,
        )
        .unwrap_err();
        assert!(
            format!("{err}").contains("only one unlimited dimension"),
            "{err}"
        );
    }

    /// A maximum shape can describe more chunks than a `u64` can number. The
    /// multiplier has to refuse rather than wrap, since a wrapped multiplier
    /// numbers two different chunks the same.
    #[test]
    fn an_unnumberable_maximum_shape_is_refused() {
        let err = ChunkGrid::new(
            &[1, 1, 1],
            &[2, 2, 2],
            Some(&[u64::MAX - 1, u64::MAX - 1, u64::MAX - 1]),
            GridOrder::RowMajor,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("can be numbered"), "{err}");
    }

    /// The slot *count* multiplies the same dimensions the multipliers do, and
    /// a rank whose multipliers all fit can still have a product that does not.
    /// It has to refuse rather than report `None`, which a Fixed Array caller
    /// reads as "unlimited, so declare no count" — the opposite of the truth.
    #[test]
    fn an_unnumberable_slot_count_is_refused_not_reported_as_unlimited() {
        let err = ChunkGrid::new(
            &[1, 1],
            &[2, 2],
            Some(&[u64::MAX - 1, u64::MAX - 1]),
            GridOrder::RowMajor,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("can be numbered"), "{err}");
    }
}
