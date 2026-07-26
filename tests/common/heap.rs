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

/// Offset of the fractal-heap header in `bytes`.
///
/// Panics if there is none, since a caller asking about heap storage has already
/// decided the file should have one.
#[track_caller]
fn frhp(bytes: &[u8]) -> usize {
    bytes
        .windows(4)
        .position(|w| w == b"FRHP")
        .expect("a dense attribute or link set has a fractal heap header")
}

/// Read a `u64` field from the fractal-heap header, `fields` 8-byte fields past
/// the fixed prefix: signature(4) + version(1) + heap ID length(2) + I/O filter
/// length(2) + flags(1) + maximum managed object size(4).
#[track_caller]
fn frhp_u64(bytes: &[u8], fields: usize) -> u64 {
    let at = frhp(bytes) + 4 + 1 + 2 + 2 + 1 + 4 + fields * SIZE;
    u64::from_le_bytes(bytes[at..at + SIZE].try_into().expect("8 bytes"))
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
