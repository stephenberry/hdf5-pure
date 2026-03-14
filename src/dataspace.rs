//! HDF5 Dataspace message parsing (message type 0x0001).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::error::FormatError;

/// Type of dataspace.
#[derive(Debug, Clone, PartialEq)]
pub enum DataspaceType {
    /// Scalar (single element).
    Scalar,
    /// Simple (N-dimensional array).
    Simple,
    /// Null (no data).
    Null,
}

/// Parsed HDF5 dataspace message.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataspace {
    /// The type of this dataspace.
    pub space_type: DataspaceType,
    /// Number of dimensions (0 for scalar).
    pub rank: u8,
    /// Current dimension sizes.
    pub dimensions: Vec<u64>,
    /// Maximum dimension sizes, if present. `u64::MAX` means unlimited.
    pub max_dimensions: Option<Vec<u64>>,
}

fn ensure_len(data: &[u8], offset: usize, needed: usize) -> Result<(), FormatError> {
    match offset.checked_add(needed) {
        Some(end) if end <= data.len() => Ok(()),
        _ => Err(FormatError::UnexpectedEof {
            expected: offset.saturating_add(needed),
            available: data.len(),
        }),
    }
}

fn read_length(data: &[u8], offset: usize, length_size: u8) -> Result<u64, FormatError> {
    let s = length_size as usize;
    ensure_len(data, offset, s)?;
    let slice = &data[offset..offset + s];
    Ok(match length_size {
        2 => u16::from_le_bytes([slice[0], slice[1]]) as u64,
        4 => u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]) as u64,
        8 => u64::from_le_bytes([
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
        ]),
        _ => {
            return Err(FormatError::InvalidLengthSize(length_size));
        }
    })
}

impl Dataspace {
    /// Parse a dataspace message from raw message bytes.
    ///
    /// `length_size` is needed for dimension value width (from superblock).
    pub fn parse(data: &[u8], length_size: u8) -> Result<Dataspace, FormatError> {
        ensure_len(data, 0, 4)?;

        let version = data[0];
        let rank = data[1];
        let flags = data[2];

        let (space_type, header_size) = match version {
            1 => {
                // v1: byte 3 is reserved, then 4 reserved bytes
                ensure_len(data, 0, 8)?;
                let st = if rank == 0 {
                    DataspaceType::Scalar
                } else {
                    DataspaceType::Simple
                };
                (st, 8usize)
            }
            2 => {
                // v2: byte 3 is type
                let type_byte = data[3];
                let st = match type_byte {
                    0 => DataspaceType::Scalar,
                    1 => DataspaceType::Simple,
                    2 => DataspaceType::Null,
                    _ => return Err(FormatError::InvalidDataspaceType(type_byte)),
                };
                (st, 4usize)
            }
            _ => return Err(FormatError::InvalidDataspaceVersion(version)),
        };

        let ls = length_size as usize;
        let mut pos = header_size;

        // Read current dimensions
        let mut dimensions = Vec::with_capacity(rank as usize);
        for _ in 0..rank {
            let dim = read_length(data, pos, length_size)?;
            dimensions.push(dim);
            pos += ls;
        }

        // Read max dimensions if flags bit 0 is set
        let max_dimensions = if flags & 0x01 != 0 {
            let mut max_dims = Vec::with_capacity(rank as usize);
            for _ in 0..rank {
                let val = read_length(data, pos, length_size)?;
                max_dims.push(val);
                pos += ls;
            }
            Some(max_dims)
        } else {
            None
        };

        // v1 flags bit 1 = permutation indices present (skip them)
        if version == 1 && flags & 0x02 != 0 {
            // rank × length_size bytes of permutation indices — skip
            let _skip = rank as usize * ls;
            // pos += _skip; // not needed since we don't use pos after this
        }

        Ok(Dataspace {
            space_type,
            rank,
            dimensions,
            max_dimensions,
        })
    }

    /// Serialize dataspace to HDF5 message bytes (v2 format).
    pub fn serialize(&self, length_size: u8) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(2); // version 2
        buf.push(self.rank);
        let flags = if self.max_dimensions.is_some() { 0x01 } else { 0x00 };
        buf.push(flags);
        let type_byte = match self.space_type {
            DataspaceType::Scalar => 0,
            DataspaceType::Simple => 1,
            DataspaceType::Null => 2,
        };
        buf.push(type_byte);
        for &dim in &self.dimensions {
            Self::write_length(&mut buf, dim, length_size);
        }
        if let Some(ref max_dims) = self.max_dimensions {
            for &md in max_dims {
                Self::write_length(&mut buf, md, length_size);
            }
        }
        buf
    }

    fn write_length(buf: &mut Vec<u8>, val: u64, size: u8) {
        match size {
            2 => buf.extend_from_slice(&(val as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&val.to_le_bytes()),
            _ => {}
        }
    }

    /// Total number of elements. Scalar = 1, Null = 0.
    pub fn num_elements(&self) -> u64 {
        match self.space_type {
            DataspaceType::Null => 0,
            DataspaceType::Scalar => 1,
            DataspaceType::Simple => {
                if self.dimensions.is_empty() {
                    0
                } else {
                    self.dimensions.iter().product()
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_v1_dataspace(rank: u8, flags: u8, dims: &[u64], max_dims: Option<&[u64]>) -> Vec<u8> {
        let length_size = 8u8;
        let mut buf = Vec::new();
        buf.push(1); // version
        buf.push(rank);
        buf.push(flags);
        buf.push(0); // reserved
        buf.extend_from_slice(&[0u8; 4]); // reserved(4)
        for &d in dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        if let Some(md) = max_dims {
            for &d in md {
                buf.extend_from_slice(&d.to_le_bytes());
            }
        }
        let _ = length_size;
        buf
    }

    fn build_v2_dataspace(
        rank: u8,
        flags: u8,
        type_byte: u8,
        dims: &[u64],
        max_dims: Option<&[u64]>,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(2); // version
        buf.push(rank);
        buf.push(flags);
        buf.push(type_byte);
        for &d in dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        if let Some(md) = max_dims {
            for &d in md {
                buf.extend_from_slice(&d.to_le_bytes());
            }
        }
        buf
    }

    #[test]
    fn scalar_v1() {
        let data = build_v1_dataspace(0, 0, &[], None);
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.space_type, DataspaceType::Scalar);
        assert_eq!(ds.rank, 0);
        assert!(ds.dimensions.is_empty());
        assert!(ds.max_dimensions.is_none());
        assert_eq!(ds.num_elements(), 1);
    }

    #[test]
    fn null_v2() {
        let data = build_v2_dataspace(0, 0, 2, &[], None);
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.space_type, DataspaceType::Null);
        assert_eq!(ds.num_elements(), 0);
    }

    #[test]
    fn simple_1d() {
        let data = build_v1_dataspace(1, 0, &[5], None);
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.space_type, DataspaceType::Simple);
        assert_eq!(ds.rank, 1);
        assert_eq!(ds.dimensions, vec![5]);
        assert!(ds.max_dimensions.is_none());
        assert_eq!(ds.num_elements(), 5);
    }

    #[test]
    fn simple_2d() {
        let data = build_v1_dataspace(2, 0, &[3, 4], None);
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.rank, 2);
        assert_eq!(ds.dimensions, vec![3, 4]);
        assert_eq!(ds.num_elements(), 12);
    }

    #[test]
    fn simple_3d_with_max_dims_unlimited() {
        let data = build_v1_dataspace(
            3,
            0x01,
            &[2, 3, 4],
            Some(&[10, u64::MAX, 100]),
        );
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.rank, 3);
        assert_eq!(ds.dimensions, vec![2, 3, 4]);
        let md = ds.max_dimensions.clone().unwrap();
        assert_eq!(md, vec![10, u64::MAX, 100]);
        assert_eq!(ds.num_elements(), 24);
    }

    #[test]
    fn v2_simple() {
        let data = build_v2_dataspace(1, 0, 1, &[7], None);
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.space_type, DataspaceType::Simple);
        assert_eq!(ds.dimensions, vec![7]);
    }

    #[test]
    fn v2_scalar() {
        let data = build_v2_dataspace(0, 0, 0, &[], None);
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.space_type, DataspaceType::Scalar);
        assert_eq!(ds.num_elements(), 1);
    }

    #[test]
    fn v1_with_4byte_length() {
        let mut buf = Vec::new();
        buf.push(1); // version
        buf.push(1); // rank
        buf.push(0); // flags
        buf.push(0); // reserved
        buf.extend_from_slice(&[0u8; 4]); // reserved(4)
        buf.extend_from_slice(&10u32.to_le_bytes()); // dim with length_size=4
        let ds = Dataspace::parse(&buf, 4).unwrap();
        assert_eq!(ds.dimensions, vec![10]);
    }

    #[test]
    fn truncated_data_error() {
        let data = [1u8, 2]; // too short
        let err = Dataspace::parse(&data, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn invalid_version_error() {
        let data = [5u8, 0, 0, 0, 0, 0, 0, 0];
        let err = Dataspace::parse(&data, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidDataspaceVersion(5));
    }

    #[test]
    fn invalid_v2_type_error() {
        let data = build_v2_dataspace(0, 0, 5, &[], None);
        let err = Dataspace::parse(&data, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidDataspaceType(5));
    }

    #[test]
    fn v1_with_max_dims() {
        let data = build_v1_dataspace(1, 0x01, &[5], Some(&[10]));
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.max_dimensions, Some(vec![10]));
    }
}
