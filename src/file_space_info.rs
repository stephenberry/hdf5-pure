//! HDF5 File Space Info message (header message type `0x0017`) and the
//! file-space management strategy it records.
//!
//! Introduced in HDF5 1.10, this message lives in the *superblock extension*
//! (a standalone object header the superblock points at) and records the
//! choices made through `H5Pset_file_space_strategy` and
//! `H5Pset_file_space_page_size`: how a file tracks and reuses free space, the
//! free-space section threshold, and the file-space page size.
//!
//! Only the version-1 layout (HDF5 1.10.1+, the only one any current tool
//! writes) is handled. Byte layout, all little-endian:
//!
//! | field                     | size        | notes                          |
//! |---------------------------|-------------|--------------------------------|
//! | version                   | 1           | always 1                       |
//! | strategy                  | 1           | [`FileSpaceStrategy`] code 0–3 |
//! | persisting free space     | 1           | 0 or 1                         |
//! | free-space threshold      | length size | smallest tracked section       |
//! | file-space page size      | length size | paged-allocation page          |
//! | page end metadata thresh. | 2           |                                |
//! | EOA before FSM allocation | offset size | `UNDEF` when not persisting     |
//! | free-space manager addrs  | offset size × N | present only when persisting |
//!
//! When free space is not persisted the manager-address array is omitted
//! entirely (a 29-byte message for the standard 8-byte sizes).

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::error::FormatError;

/// An undefined on-disk address (all bits set), HDF5's "no address" sentinel.
const UNDEF: u64 = u64::MAX;

/// The default free-space section threshold the C library uses.
pub(crate) const DEFAULT_THRESHOLD: u64 = 1;
/// The default file-space page size the C library uses.
pub(crate) const DEFAULT_PAGE_SIZE: u64 = 4096;
/// Number of free-space-manager address slots a persisting message carries (one
/// per file memory type); the reference C library writes twelve.
pub(crate) const NUM_FILE_FSM_MANAGERS: usize = 12;

/// File-space management strategy, mirroring HDF5's `H5F_fspace_strategy_t`
/// (set with `H5Pset_file_space_strategy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileSpaceStrategy {
    /// Free-space managers, aggregators, and the virtual file driver — the
    /// HDF5 default. `H5F_FSPACE_STRATEGY_FSM_AGGR`.
    FsmAggr,
    /// Paged aggregation backed by free-space managers.
    /// `H5F_FSPACE_STRATEGY_PAGE`.
    Page,
    /// Aggregators and the virtual file driver only, no free-space managers.
    /// `H5F_FSPACE_STRATEGY_AGGR`.
    Aggr,
    /// No free-space tracking; allocation only ever appends.
    /// `H5F_FSPACE_STRATEGY_NONE`.
    None,
}

impl FileSpaceStrategy {
    /// The on-disk numeric code (0–3).
    pub(crate) fn to_code(self) -> u8 {
        match self {
            FileSpaceStrategy::FsmAggr => 0,
            FileSpaceStrategy::Page => 1,
            FileSpaceStrategy::Aggr => 2,
            FileSpaceStrategy::None => 3,
        }
    }

    fn from_code(code: u8) -> Result<Self, FormatError> {
        match code {
            0 => Ok(FileSpaceStrategy::FsmAggr),
            1 => Ok(FileSpaceStrategy::Page),
            2 => Ok(FileSpaceStrategy::Aggr),
            3 => Ok(FileSpaceStrategy::None),
            other => Err(FormatError::InvalidFileSpaceStrategy(other)),
        }
    }
}

/// A parsed (or to-be-written) File Space Info message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSpaceInfo {
    /// The file-space management strategy.
    pub strategy: FileSpaceStrategy,
    /// Whether free space is persisted to disk across file close (the manager
    /// addresses below are written only when this is set).
    pub persist: bool,
    /// Smallest free-space section size the managers track.
    pub threshold: u64,
    /// File-space page size used for paged allocation.
    pub page_size: u64,
    /// Page-end metadata threshold (paged allocation tuning).
    pub page_end_meta_threshold: u16,
    /// End-of-allocation address recorded before free-space manager metadata
    /// was allocated; [`u64::MAX`] when free space is not persisted.
    pub eoa_pre_fsm: u64,
    /// Free-space manager header addresses (present only when [`persist`] is
    /// set); unused slots are [`u64::MAX`]. Followed to their on-disk `FSHD`/
    /// `FSSE` blocks by [`File::persisted_free_space`](crate::File::persisted_free_space).
    ///
    /// [`persist`]: Self::persist
    pub manager_addrs: Vec<u64>,
}

impl FileSpaceInfo {
    /// A non-persisting message recording `strategy` with the given thresholds.
    /// This is the form the writer emits (no free-space manager blocks).
    pub(crate) fn non_persistent(
        strategy: FileSpaceStrategy,
        threshold: u64,
        page_size: u64,
    ) -> Self {
        FileSpaceInfo {
            strategy,
            persist: false,
            threshold,
            page_size,
            page_end_meta_threshold: 0,
            eoa_pre_fsm: UNDEF,
            manager_addrs: Vec::new(),
        }
    }

    /// A persisting message for a file with no free space yet (the form
    /// [`FileBuilder`](crate::FileBuilder) emits for `persist = true`): the
    /// persist flag is set but every manager slot is undefined and no FSM space
    /// has been allocated, so `eoa_pre_fsm` is [`UNDEF`]. This matches what the C
    /// library records when persistence is on but nothing is tracked.
    pub(crate) fn persistent_empty(
        strategy: FileSpaceStrategy,
        threshold: u64,
        page_size: u64,
    ) -> Self {
        FileSpaceInfo {
            strategy,
            persist: true,
            threshold,
            page_size,
            page_end_meta_threshold: 0,
            eoa_pre_fsm: UNDEF,
            manager_addrs: vec![UNDEF; NUM_FILE_FSM_MANAGERS],
        }
    }

    /// A persisting message whose first free-space manager is at `manager0_addr`
    /// (the others undefined), recording `eoa_pre_fsm` — the end-of-allocation
    /// before the on-disk free-space-manager blocks were appended. This is the
    /// form [`EditSession`](crate::EditSession) writes when it persists a non-empty
    /// free list: every tracked region lives in that one manager.
    pub(crate) fn persistent_single_manager(
        strategy: FileSpaceStrategy,
        threshold: u64,
        page_size: u64,
        manager0_addr: u64,
        eoa_pre_fsm: u64,
    ) -> Self {
        let mut manager_addrs = vec![UNDEF; NUM_FILE_FSM_MANAGERS];
        manager_addrs[0] = manager0_addr;
        FileSpaceInfo {
            strategy,
            persist: true,
            threshold,
            page_size,
            page_end_meta_threshold: 0,
            eoa_pre_fsm,
            manager_addrs,
        }
    }

    /// A persisting message for a paged file whose free space is tracked by
    /// per-page-type managers. `slots[k]` is the `FSHD` address of the manager for
    /// page type `k + 1` (or [`UNDEF`] when that page type tracks no free space);
    /// `eoa_pre_fsm` is the page-aligned end-of-allocation. This is the form the
    /// paged writer emits: metadata free space lives in the SUPER manager
    /// (`slots[0]`), small raw data in DRAW (`slots[2]`), and the trailing
    /// fragments of large multi-page allocations in the generic-large manager
    /// (`slots[6]`).
    pub(crate) fn persistent_managers(
        strategy: FileSpaceStrategy,
        threshold: u64,
        page_size: u64,
        slots: [u64; NUM_FILE_FSM_MANAGERS],
        eoa_pre_fsm: u64,
    ) -> Self {
        FileSpaceInfo {
            strategy,
            persist: true,
            threshold,
            page_size,
            page_end_meta_threshold: 0,
            eoa_pre_fsm,
            manager_addrs: slots.to_vec(),
        }
    }

    /// Serialize the version-1 message body (without the object-header message
    /// prefix). Manager addresses are written only when [`persist`] is set.
    ///
    /// [`persist`]: Self::persist
    pub(crate) fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(29 + self.manager_addrs.len() * 8);
        buf.push(1); // version
        buf.push(self.strategy.to_code());
        buf.push(self.persist as u8);
        buf.extend_from_slice(&self.threshold.to_le_bytes());
        buf.extend_from_slice(&self.page_size.to_le_bytes());
        buf.extend_from_slice(&self.page_end_meta_threshold.to_le_bytes());
        buf.extend_from_slice(&self.eoa_pre_fsm.to_le_bytes());
        if self.persist {
            for &addr in &self.manager_addrs {
                buf.extend_from_slice(&addr.to_le_bytes());
            }
        }
        buf
    }

    /// Parse a version-1 message body. `offset_size`/`length_size` come from the
    /// superblock (both 8 for standard files).
    pub(crate) fn parse(
        data: &[u8],
        offset_size: u8,
        length_size: u8,
    ) -> Result<FileSpaceInfo, FormatError> {
        let os = offset_size as usize;
        let ls = length_size as usize;
        // version(1) + strategy(1) + persist(1) + threshold(ls) + page_size(ls)
        // + page_end(2) + eoa(os)
        let fixed = 3 + ls + ls + 2 + os;
        if data.len() < fixed {
            return Err(FormatError::UnexpectedEof {
                expected: fixed,
                available: data.len(),
            });
        }
        let version = data[0];
        if version != 1 {
            return Err(FormatError::UnsupportedFileSpaceInfoVersion(version));
        }
        let strategy = FileSpaceStrategy::from_code(data[1])?;
        let persist = data[2] != 0;
        let mut pos = 3;
        let threshold = read_uint_le(&data[pos..pos + ls]);
        pos += ls;
        let page_size = read_uint_le(&data[pos..pos + ls]);
        pos += ls;
        let page_end_meta_threshold = u16::from_le_bytes([data[pos], data[pos + 1]]);
        pos += 2;
        let eoa_pre_fsm = read_uint_le(&data[pos..pos + os]);
        pos += os;

        let mut manager_addrs = Vec::new();
        if persist {
            // The remaining bytes are offset-size manager addresses; read as
            // many as are present rather than assuming a fixed count.
            while pos + os <= data.len() {
                manager_addrs.push(read_uint_le(&data[pos..pos + os]));
                pos += os;
            }
        }

        Ok(FileSpaceInfo {
            strategy,
            persist,
            threshold,
            page_size,
            page_end_meta_threshold,
            eoa_pre_fsm,
            manager_addrs,
        })
    }
}

/// Read a little-endian unsigned integer of 1–8 bytes into a `u64`.
fn read_uint_le(bytes: &[u8]) -> u64 {
    let mut v = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        v |= (b as u64) << (8 * i);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_persistent_roundtrip_29_bytes() {
        for strategy in [
            FileSpaceStrategy::FsmAggr,
            FileSpaceStrategy::Page,
            FileSpaceStrategy::Aggr,
            FileSpaceStrategy::None,
        ] {
            let info = FileSpaceInfo::non_persistent(strategy, 1, 4096);
            let bytes = info.serialize();
            assert_eq!(bytes.len(), 29, "non-persistent message is 29 bytes");
            let parsed = FileSpaceInfo::parse(&bytes, 8, 8).unwrap();
            assert_eq!(parsed, info);
            assert_eq!(parsed.eoa_pre_fsm, u64::MAX);
            assert!(parsed.manager_addrs.is_empty());
        }
    }

    #[test]
    fn matches_c_library_none_bytes() {
        // Exact bytes the reference C library (HDF5 1.14.6) wrote for strategy
        // NONE, captured via tmp/probe_fsinfo.py.
        let expected = [
            0x01u8, 0x03, 0x00, // version=1, strategy=NONE(3), persist=0
            0x01, 0, 0, 0, 0, 0, 0, 0, // threshold=1
            0x00, 0x10, 0, 0, 0, 0, 0, 0, // page_size=4096
            0x00, 0x00, // page end meta threshold
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // eoa = UNDEF
        ];
        let info = FileSpaceInfo::non_persistent(FileSpaceStrategy::None, 1, 4096);
        assert_eq!(info.serialize(), expected);
    }

    #[test]
    fn parses_persistent_manager_addresses() {
        // A persisting message: 29-byte head + three 8-byte manager addresses.
        let mut bytes = FileSpaceInfo {
            strategy: FileSpaceStrategy::FsmAggr,
            persist: true,
            threshold: 1,
            page_size: 4096,
            page_end_meta_threshold: 0,
            eoa_pre_fsm: 2072,
            manager_addrs: vec![619, u64::MAX, u64::MAX],
        }
        .serialize();
        assert_eq!(bytes.len(), 29 + 3 * 8);
        let parsed = FileSpaceInfo::parse(&bytes, 8, 8).unwrap();
        assert_eq!(parsed.manager_addrs, vec![619, u64::MAX, u64::MAX]);
        assert_eq!(parsed.eoa_pre_fsm, 2072);
        assert!(parsed.persist);

        // Truncate the version byte to an unsupported value -> clean error.
        bytes[0] = 0;
        assert!(matches!(
            FileSpaceInfo::parse(&bytes, 8, 8),
            Err(FormatError::UnsupportedFileSpaceInfoVersion(0))
        ));
    }

    #[test]
    fn persistent_managers_roundtrips_multiple_slots() {
        // The paged writer's form: SUPER (slot 0), DRAW (slot 2), and the
        // generic-large manager (slot 6) defined, the rest undefined.
        let mut slots = [UNDEF; NUM_FILE_FSM_MANAGERS];
        slots[0] = 841;
        slots[2] = 18384;
        slots[6] = 806;
        let info =
            FileSpaceInfo::persistent_managers(FileSpaceStrategy::Page, 0, 16384, slots, 65536);
        assert!(info.persist);
        assert_eq!(info.eoa_pre_fsm, 65536);
        let bytes = info.serialize();
        // 29-byte head + 12 * 8 manager slots.
        assert_eq!(bytes.len(), 29 + NUM_FILE_FSM_MANAGERS * 8);
        let parsed = FileSpaceInfo::parse(&bytes, 8, 8).unwrap();
        assert_eq!(parsed, info);
        assert_eq!(parsed.manager_addrs[0], 841);
        assert_eq!(parsed.manager_addrs[2], 18384);
        assert_eq!(parsed.manager_addrs[6], 806);
        assert_eq!(parsed.manager_addrs[1], UNDEF);
    }

    #[test]
    fn rejects_bad_strategy_code() {
        assert!(matches!(
            FileSpaceStrategy::from_code(4),
            Err(FormatError::InvalidFileSpaceStrategy(4))
        ));
    }
}
