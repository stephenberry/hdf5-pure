//! HDF5 file signature (magic bytes) detection.

use crate::error::FormatError;

/// The 8-byte HDF5 magic signature.
pub const HDF5_SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', b'\r', b'\n', 0x1A, b'\n'];

/// Search for the HDF5 signature at valid offsets.
///
/// The HDF5 spec says the signature can appear at offset 0, 512, 1024, 2048, 4096, ...
/// (powers of two starting at 512, plus offset 0).
///
/// Returns the byte offset where the signature was found.
pub fn find_signature(data: &[u8]) -> Result<usize, FormatError> {
    // Check offset 0
    if data.len() >= 8 && data[..8] == HDF5_SIGNATURE {
        return Ok(0);
    }

    // Check powers of 2 starting at 512
    let mut offset = 512;
    while offset + 8 <= data.len() {
        if data[offset..offset + 8] == HDF5_SIGNATURE {
            return Ok(offset);
        }
        offset *= 2;
    }

    Err(FormatError::SignatureNotFound)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_at_offset_0() {
        let mut data = vec![0u8; 64];
        data[..8].copy_from_slice(&HDF5_SIGNATURE);
        assert_eq!(find_signature(&data), Ok(0));
    }

    #[test]
    fn signature_at_offset_512() {
        let mut data = vec![0u8; 1024];
        data[512..520].copy_from_slice(&HDF5_SIGNATURE);
        assert_eq!(find_signature(&data), Ok(512));
    }

    #[test]
    fn signature_at_offset_1024() {
        let mut data = vec![0u8; 2048];
        data[1024..1032].copy_from_slice(&HDF5_SIGNATURE);
        assert_eq!(find_signature(&data), Ok(1024));
    }

    #[test]
    fn signature_at_offset_2048() {
        let mut data = vec![0u8; 4096];
        data[2048..2056].copy_from_slice(&HDF5_SIGNATURE);
        assert_eq!(find_signature(&data), Ok(2048));
    }

    #[test]
    fn signature_not_found() {
        let data = vec![0u8; 8192];
        assert_eq!(find_signature(&data), Err(FormatError::SignatureNotFound));
    }

    #[test]
    fn signature_not_found_empty() {
        assert_eq!(find_signature(&[]), Err(FormatError::SignatureNotFound));
    }

    #[test]
    fn signature_not_found_too_short() {
        assert_eq!(
            find_signature(&[0x89, b'H', b'D']),
            Err(FormatError::SignatureNotFound)
        );
    }

    #[test]
    fn signature_at_non_power_of_two_not_found() {
        // Signature at offset 100 should NOT be found
        let mut data = vec![0u8; 1024];
        data[100..108].copy_from_slice(&HDF5_SIGNATURE);
        assert_eq!(find_signature(&data), Err(FormatError::SignatureNotFound));
    }

    #[test]
    fn signature_prefers_earliest() {
        // Signature at both 0 and 512, should return 0
        let mut data = vec![0u8; 1024];
        data[..8].copy_from_slice(&HDF5_SIGNATURE);
        data[512..520].copy_from_slice(&HDF5_SIGNATURE);
        assert_eq!(find_signature(&data), Ok(0));
    }
}
