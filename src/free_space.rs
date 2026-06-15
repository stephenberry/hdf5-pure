//! Session-local free-space tracking for in-place editing (issue #21).
//!
//! [`EditSession`](crate::EditSession) writes by appending at end-of-file and,
//! on each commit, leaves the superseded object headers and any deleted-object
//! blocks behind as dead bytes. This module records those freed regions so a
//! later allocation can reuse them instead of growing the file, and so a run of
//! free space that reaches end-of-file can be truncated away.
//!
//! It is the in-memory half of HDF5's "free-space management". For a file opened
//! without persistence (the default) it is purely session-local: freed-but-
//! unreused space is invisible to other tools, exactly as the reference C
//! library's default `FSM_AGGR` strategy with persistence off leaves it. When the
//! file was created with `persist = true`, [`EditSession`](crate::EditSession)
//! seeds this list from the on-disk free-space managers (the `FSHD`/`FSSE` blocks
//! the File Space Info superblock-extension message points at) on open and writes
//! it back on each commit, so reuse spans sessions (see
//! [`free_space_manager`](crate::free_space_manager)).
//!
//! The structure is a sorted, fully coalesced list of disjoint `[addr, addr+len)`
//! regions. Every public operation preserves both invariants (sorted by address,
//! no two regions touching or overlapping), so the list is always in a canonical
//! form and `trailing_free` is a single comparison against the highest region.

/// A contiguous run of free bytes in the file, `[addr, addr + len)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FreeRegion {
    addr: u64,
    len: u64,
}

impl FreeRegion {
    /// One past the last byte of the region.
    fn end(&self) -> u64 {
        self.addr + self.len
    }
}

/// A sorted, coalesced set of free regions in a single file being edited.
///
/// Invariants, upheld by every method: regions are sorted by `addr`, are
/// non-empty, and never touch or overlap (any two that would are merged on
/// insertion). Allocation is best-fit to limit fragmentation.
#[derive(Debug, Default, Clone)]
pub(crate) struct FreeList {
    /// Disjoint regions, sorted ascending by address and never adjacent.
    regions: Vec<FreeRegion>,
}

impl FreeList {
    /// An empty free list.
    pub(crate) fn new() -> Self {
        Self {
            regions: Vec::new(),
        }
    }

    /// Record `[addr, addr + len)` as free, merging it with any adjacent or
    /// overlapping regions so the list stays canonical.
    ///
    /// A zero-length free is a no-op. Overlapping an already-free region is a
    /// caller bug (a double-free): in debug builds it panics; in release builds
    /// the overlap is absorbed by the merge rather than corrupting the list.
    pub(crate) fn free(&mut self, addr: u64, len: u64) {
        if len == 0 {
            return;
        }
        let new_end = addr + len;

        // Find the first region that ends at or after `addr` — the leftmost one
        // that could touch or overlap the freed range. Everything before it is
        // strictly to the left with a gap and stays untouched.
        let mut lo = 0;
        while lo < self.regions.len() && self.regions[lo].end() < addr {
            lo += 1;
        }

        // Find the end of the run of regions that touch or overlap `[addr,
        // new_end)`: any region whose start is <= new_end is adjacent/overlapping
        // and folds into the merged region.
        let mut hi = lo;
        let mut merged_addr = addr;
        let mut merged_end = new_end;
        while hi < self.regions.len() && self.regions[hi].addr <= merged_end {
            debug_assert!(
                self.regions[hi].addr >= new_end || self.regions[hi].end() <= addr,
                "double-free: [{addr}, {new_end}) overlaps free region [{}, {})",
                self.regions[hi].addr,
                self.regions[hi].end()
            );
            merged_addr = merged_addr.min(self.regions[hi].addr);
            merged_end = merged_end.max(self.regions[hi].end());
            hi += 1;
        }

        let merged = FreeRegion {
            addr: merged_addr,
            len: merged_end - merged_addr,
        };
        self.regions.splice(lo..hi, [merged]);
    }

    /// Reserve `len` bytes from a free region, returning the address handed out,
    /// or `None` if no single region is large enough.
    ///
    /// Best-fit: the smallest region that fits, to keep large runs intact. The
    /// allocation is taken from the low end of the chosen region; any remainder
    /// stays free. `len` of 0 returns `None` (nothing to allocate).
    pub(crate) fn alloc(&mut self, len: u64) -> Option<u64> {
        if len == 0 {
            return None;
        }
        let mut best: Option<usize> = None;
        for (i, r) in self.regions.iter().enumerate() {
            if r.len >= len && best.is_none_or(|b| r.len < self.regions[b].len) {
                best = Some(i);
            }
        }
        let i = best?;
        let addr = self.regions[i].addr;
        if self.regions[i].len == len {
            self.regions.remove(i);
        } else {
            self.regions[i].addr += len;
            self.regions[i].len -= len;
        }
        Some(addr)
    }

    /// The free regions as `(addr, len)` pairs, sorted ascending by address and
    /// fully coalesced. Used to persist the free list to disk (issue #21).
    pub(crate) fn sections(&self) -> Vec<(u64, u64)> {
        self.regions.iter().map(|r| (r.addr, r.len)).collect()
    }

    /// If a free region ends exactly at `eof` (the current end-of-file), remove
    /// it from the list and return its start address — the file can be truncated
    /// to that address. Returns `None` if the highest free region does not reach
    /// end-of-file.
    ///
    /// Because the list is coalesced, at most one region can end at `eof`, and it
    /// is the last one.
    pub(crate) fn take_trailing(&mut self, eof: u64) -> Option<u64> {
        match self.regions.last() {
            Some(last) if last.end() == eof => {
                let addr = last.addr;
                self.regions.pop();
                Some(addr)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expose the canonical region list as `(addr, len)` pairs for assertions.
    fn regions(fl: &FreeList) -> Vec<(u64, u64)> {
        fl.regions.iter().map(|r| (r.addr, r.len)).collect()
    }

    #[test]
    fn free_into_empty_list() {
        let mut fl = FreeList::new();
        fl.free(100, 50);
        assert_eq!(regions(&fl), [(100, 50)]);
    }

    #[test]
    fn zero_length_free_is_noop() {
        let mut fl = FreeList::new();
        fl.free(100, 0);
        assert!(regions(&fl).is_empty());
    }

    #[test]
    fn disjoint_frees_stay_sorted() {
        let mut fl = FreeList::new();
        fl.free(300, 10);
        fl.free(100, 10);
        fl.free(200, 10);
        assert_eq!(regions(&fl), [(100, 10), (200, 10), (300, 10)]);
    }

    #[test]
    fn coalesce_with_right_neighbor() {
        let mut fl = FreeList::new();
        fl.free(200, 50); // [200, 250)
        fl.free(150, 50); // [150, 200) touches left edge of the above
        assert_eq!(regions(&fl), [(150, 100)]);
    }

    #[test]
    fn coalesce_with_left_neighbor() {
        let mut fl = FreeList::new();
        fl.free(150, 50); // [150, 200)
        fl.free(200, 50); // [200, 250) touches right edge of the above
        assert_eq!(regions(&fl), [(150, 100)]);
    }

    #[test]
    fn coalesce_bridges_gap_between_two() {
        let mut fl = FreeList::new();
        fl.free(100, 50); // [100, 150)
        fl.free(250, 50); // [250, 300)
        fl.free(150, 100); // [150, 250) bridges the two
        assert_eq!(regions(&fl), [(100, 200)]);
    }

    #[test]
    fn no_coalesce_when_gap_remains() {
        let mut fl = FreeList::new();
        fl.free(100, 50); // [100, 150)
        fl.free(151, 50); // [151, 201) one byte gap
        assert_eq!(regions(&fl), [(100, 50), (151, 50)]);
    }

    #[test]
    fn alloc_best_fit_chooses_smallest_sufficient() {
        let mut fl = FreeList::new();
        fl.free(0, 100); // big
        fl.free(200, 30); // exact-ish, smallest that fits 30
        fl.free(400, 60); // medium
        let addr = fl.alloc(30).unwrap();
        assert_eq!(addr, 200);
        // The 30-region is consumed exactly; the others remain.
        assert_eq!(regions(&fl), [(0, 100), (400, 60)]);
    }

    #[test]
    fn alloc_splits_remainder() {
        let mut fl = FreeList::new();
        fl.free(1000, 100);
        let addr = fl.alloc(40).unwrap();
        assert_eq!(addr, 1000);
        assert_eq!(regions(&fl), [(1040, 60)]);
    }

    #[test]
    fn alloc_none_when_nothing_fits() {
        let mut fl = FreeList::new();
        fl.free(0, 10);
        fl.free(100, 20);
        assert!(fl.alloc(50).is_none());
        // List is unchanged on a failed allocation.
        assert_eq!(regions(&fl), [(0, 10), (100, 20)]);
    }

    #[test]
    fn alloc_zero_returns_none() {
        let mut fl = FreeList::new();
        fl.free(0, 100);
        assert!(fl.alloc(0).is_none());
    }

    #[test]
    fn alloc_then_free_roundtrips() {
        let mut fl = FreeList::new();
        fl.free(0, 100);
        let a = fl.alloc(40).unwrap();
        fl.free(a, 40); // give it back
        assert_eq!(regions(&fl), [(0, 100)]); // coalesced back to whole
    }

    #[test]
    fn take_trailing_at_eof() {
        let mut fl = FreeList::new();
        fl.free(500, 100); // [500, 600)
        let cut = fl.take_trailing(600);
        assert_eq!(cut, Some(500));
        assert!(regions(&fl).is_empty());
    }

    #[test]
    fn take_trailing_none_when_not_at_eof() {
        let mut fl = FreeList::new();
        fl.free(500, 100); // [500, 600)
        assert_eq!(fl.take_trailing(700), None); // live bytes between 600 and 700
        assert_eq!(regions(&fl), [(500, 100)]); // unchanged
    }

    #[test]
    fn take_trailing_only_cuts_the_tail_region() {
        let mut fl = FreeList::new();
        fl.free(100, 50); // interior hole [100, 150)
        fl.free(500, 100); // trailing [500, 600)
        let cut = fl.take_trailing(600);
        assert_eq!(cut, Some(500));
        assert_eq!(regions(&fl), [(100, 50)]); // interior hole preserved
    }

    #[test]
    fn take_trailing_empty_list() {
        let mut fl = FreeList::new();
        assert_eq!(fl.take_trailing(0), None);
    }
}
