//! Reading fractal-heap facts back out of a written file, from the outside.
//!
//! Round-tripping an attribute proves it survived; it does not prove *how* it was
//! stored. These helpers decode the heap header directly so a test can assert the
//! storage class the emitter chose, which is the thing a managed/huge split is
//! actually about.
//!
//! Pure Rust on purpose: pinned here rather than in each test so the two callers
//! cannot drift on the header layout, and so 32-bit targets (which have no
//! reference C library) can still use it.
#![allow(dead_code)]

/// Byte offsets are for the 8-byte offset/length sizes this writer emits.
const SIZE: usize = 8;

/// Whether the file contains a fractal heap at all — the signature of dense
/// (heap) rather than compact (in-object-header) storage.
pub fn has_fractal_heap(bytes: &[u8]) -> bool {
    bytes.windows(4).any(|w| w == b"FRHP")
}

/// Offsets of every fractal-heap header in `bytes`, in file order.
pub fn frhp_offsets(bytes: &[u8]) -> Vec<usize> {
    bytes
        .windows(4)
        .enumerate()
        .filter(|(_, w)| *w == b"FRHP")
        .map(|(at, _)| at)
        .collect()
}

/// Offset of the first fractal-heap header in `bytes`.
///
/// Panics if there is none, since a caller asking about heap storage has already
/// decided the file should have one.
#[track_caller]
fn frhp(bytes: &[u8]) -> usize {
    *frhp_offsets(bytes)
        .first()
        .expect("a dense attribute or link set has a fractal heap header")
}

/// Read a `u64` field from the fractal-heap header at `frhp`, `fields` 8-byte
/// fields past the fixed prefix: signature(4) + version(1) + heap ID length(2) +
/// I/O filter length(2) + flags(1) + maximum managed object size(4).
#[track_caller]
fn frhp_u64_at(bytes: &[u8], frhp: usize, fields: usize) -> u64 {
    let at = frhp + 4 + 1 + 2 + 2 + 1 + 4 + fields * SIZE;
    u64::from_le_bytes(bytes[at..at + SIZE].try_into().expect("8 bytes"))
}

/// Read a `u64` field from the first fractal-heap header in `bytes`.
#[track_caller]
fn frhp_u64(bytes: &[u8], fields: usize) -> u64 {
    frhp_u64_at(bytes, frhp(bytes), fields)
}

/// How many objects the heap stores as fractal-heap *huge* objects — held outside
/// the managed direct blocks and indexed by the huge-objects v2 B-tree.
///
/// The header's field order is next huge object ID, huge B-tree address, free
/// space, free-space manager address, managed space, allocated managed space,
/// allocation iterator, managed object count, huge objects size, then this.
#[track_caller]
pub fn huge_object_count(bytes: &[u8]) -> u64 {
    frhp_u64(bytes, 9)
}

/// [`huge_object_count`] for every heap in the file, in file order — for asserting
/// what a *copy's* heap chose, which the first-heap reader cannot see.
#[track_caller]
pub fn huge_object_counts(bytes: &[u8]) -> Vec<u64> {
    frhp_offsets(bytes)
        .into_iter()
        .map(|at| frhp_u64_at(bytes, at, 9))
        .collect()
}

/// How many objects the heap stores as managed objects, inside its direct blocks.
#[track_caller]
pub fn managed_object_count(bytes: &[u8]) -> u64 {
    frhp_u64(bytes, 7)
}

/// The total byte size the heap declares for its huge objects.
#[track_caller]
pub fn huge_object_bytes(bytes: &[u8]) -> u64 {
    frhp_u64(bytes, 8)
}

/// The heap's "current # of rows in root indirect block": 0 when the root is a
/// single direct block, and otherwise how many doubling-table rows the root
/// indirect block spans.
///
/// Past the twelve 8-byte fields [`frhp_u64_at`] indexes come the doubling-table
/// fields — table width(2), starting block size(8), maximum direct block size(8),
/// maximum heap size(2), starting root rows(2) — then the root block address(8)
/// and this.
#[track_caller]
pub fn root_indirect_rows(bytes: &[u8]) -> u16 {
    let at = frhp(bytes) + 4 + 1 + 2 + 2 + 1 + 4 + 12 * SIZE + 2 + SIZE + SIZE + 2 + 2 + SIZE;
    u16::from_le_bytes(bytes[at..at + 2].try_into().expect("2 bytes"))
}

/// How many fractal-heap indirect blocks the file holds. More than one means the
/// root's own row of them filled up and the table nested.
pub fn indirect_block_count(bytes: &[u8]) -> usize {
    bytes.windows(4).filter(|w| *w == b"FHIB").count()
}
