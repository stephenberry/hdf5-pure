//! HDF5 filter implementations: deflate, shuffle, fletcher32.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::error::FormatError;
use crate::filter_pipeline::{FilterPipeline, FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_SHUFFLE};

/// Apply a filter pipeline to decompress a chunk.
/// Filters are applied in REVERSE order for decompression.
pub fn decompress_chunk(
    compressed: &[u8],
    pipeline: &FilterPipeline,
    _chunk_size: usize,
    element_size: u32,
) -> Result<Vec<u8>, FormatError> {
    let mut data = compressed.to_vec();

    for filter in pipeline.filters.iter().rev() {
        data = match filter.filter_id {
            FILTER_SHUFFLE => shuffle_decompress(&data, element_size as usize)?,
            FILTER_DEFLATE => deflate_decompress(&data)?,
            FILTER_FLETCHER32 => fletcher32_verify(&data)?,
            other => return Err(FormatError::UnsupportedFilter(other)),
        };
    }

    Ok(data)
}

/// Apply a filter pipeline to compress a chunk.
/// Filters are applied in FORWARD order for compression.
pub fn compress_chunk(
    data: &[u8],
    pipeline: &FilterPipeline,
    element_size: u32,
) -> Result<Vec<u8>, FormatError> {
    let mut result = data.to_vec();

    for filter in &pipeline.filters {
        result = match filter.filter_id {
            FILTER_SHUFFLE => shuffle_compress(&result, element_size as usize)?,
            FILTER_DEFLATE => {
                let level = filter.client_data.first().copied().unwrap_or(6);
                deflate_compress(&result, level)?
            }
            FILTER_FLETCHER32 => fletcher32_append(&result)?,
            other => return Err(FormatError::UnsupportedFilter(other)),
        };
    }

    Ok(result)
}

/// Decompress zlib-compressed data.
#[cfg(feature = "deflate")]
fn deflate_decompress(data: &[u8]) -> Result<Vec<u8>, FormatError> {
    use std::io::Read;
    let mut decoder = flate2::read::ZlibDecoder::new(data);
    let mut result = Vec::new();
    decoder
        .read_to_end(&mut result)
        .map_err(|e| FormatError::DecompressionError(e.to_string()))?;
    Ok(result)
}

#[cfg(not(feature = "deflate"))]
fn deflate_decompress(_data: &[u8]) -> Result<Vec<u8>, FormatError> {
    Err(FormatError::UnsupportedFilter(FILTER_DEFLATE))
}

/// Compress data with zlib.
#[cfg(feature = "deflate")]
fn deflate_compress(data: &[u8], level: u32) -> Result<Vec<u8>, FormatError> {
    use std::io::Write;
    let mut encoder = flate2::write::ZlibEncoder::new(
        Vec::new(),
        flate2::Compression::new(level),
    );
    encoder
        .write_all(data)
        .map_err(|e| FormatError::CompressionError(e.to_string()))?;
    encoder
        .finish()
        .map_err(|e| FormatError::CompressionError(e.to_string()))
}

#[cfg(not(feature = "deflate"))]
fn deflate_compress(_data: &[u8], _level: u32) -> Result<Vec<u8>, FormatError> {
    Err(FormatError::UnsupportedFilter(FILTER_DEFLATE))
}

/// Unshuffle (decompress direction): reconstruct interleaved element bytes.
/// On disk: all byte-0s of each element together, then all byte-1s, etc.
/// Output: elements in natural order.
fn shuffle_decompress(data: &[u8], element_size: usize) -> Result<Vec<u8>, FormatError> {
    if element_size <= 1 {
        return Ok(data.to_vec());
    }
    if !data.len().is_multiple_of(element_size) {
        return Err(FormatError::FilterError(
            "shuffle: data length not a multiple of element size".into(),
        ));
    }
    let num_elements = data.len() / element_size;
    let mut result = vec![0u8; data.len()];

    for i in 0..num_elements {
        for j in 0..element_size {
            result[i * element_size + j] = data[j * num_elements + i];
        }
    }

    Ok(result)
}

/// Shuffle (compress direction): group bytes by position within each element.
fn shuffle_compress(data: &[u8], element_size: usize) -> Result<Vec<u8>, FormatError> {
    if element_size <= 1 {
        return Ok(data.to_vec());
    }
    if !data.len().is_multiple_of(element_size) {
        return Err(FormatError::FilterError(
            "shuffle: data length not a multiple of element size".into(),
        ));
    }
    let num_elements = data.len() / element_size;
    let mut result = vec![0u8; data.len()];

    for i in 0..num_elements {
        for j in 0..element_size {
            result[j * num_elements + i] = data[i * element_size + j];
        }
    }

    Ok(result)
}

/// Compute HDF5 Fletcher32 checksum over data.
/// HDF5 uses a modified Fletcher32 that operates on 16-bit words.
///
/// Optimized with wider accumulators: processes blocks of 360 words before
/// taking the modulo, reducing the number of expensive modulo operations.
/// (360 is the maximum block size that avoids u32 overflow for sum2.)
fn fletcher32_compute(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0;
    let mut sum2: u32 = 0;

    // Process in blocks of 360 16-bit words (720 bytes) to delay modulo.
    // Max sum1 before mod: 360 * 65535 = 23_592_600 < u32::MAX
    // Max sum2 before mod: 360 * 23_592_600 ~ 8.5B > u32::MAX, but actual
    // sum2 accumulates incrementally, so worst case is 360*360*65535/2 which
    // fits in u64. We use u32 with block size 360 which is safe.
    const BLOCK_WORDS: usize = 360;
    const BLOCK_BYTES: usize = BLOCK_WORDS * 2;

    let mut offset = 0;
    let len = data.len();

    while offset + BLOCK_BYTES <= len {
        let end = offset + BLOCK_BYTES;
        let mut i = offset;
        while i < end {
            let val = ((data[i] as u32) << 8) | (data[i + 1] as u32);
            sum1 += val;
            sum2 += sum1;
            i += 2;
        }
        sum1 %= 65535;
        sum2 %= 65535;
        offset = end;
    }

    // Handle remaining bytes
    while offset < len {
        let val = if offset + 1 < len {
            ((data[offset] as u32) << 8) | (data[offset + 1] as u32)
        } else {
            (data[offset] as u32) << 8
        };
        sum1 = (sum1 + val) % 65535;
        sum2 = (sum2 + sum1) % 65535;
        offset += 2;
    }

    (sum2 << 16) | sum1
}

/// Verify Fletcher32 checksum and strip it from the data.
/// The last 4 bytes are the stored checksum.
fn fletcher32_verify(data: &[u8]) -> Result<Vec<u8>, FormatError> {
    if data.len() < 4 {
        return Err(FormatError::FilterError(
            "fletcher32: data too short for checksum".into(),
        ));
    }
    let payload = &data[..data.len() - 4];
    let stored = u32::from_le_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);
    let computed = fletcher32_compute(payload);
    if stored != computed {
        return Err(FormatError::Fletcher32Mismatch {
            expected: stored,
            computed,
        });
    }
    Ok(payload.to_vec())
}

/// Append Fletcher32 checksum to data.
fn fletcher32_append(data: &[u8]) -> Result<Vec<u8>, FormatError> {
    let checksum = fletcher32_compute(data);
    let mut result = data.to_vec();
    result.extend_from_slice(&checksum.to_le_bytes());
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_pipeline::FilterDescription;

    // --- Deflate tests ---

    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_compress_decompress_roundtrip() {
        let data: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
        let compressed = deflate_compress(&data, 6).unwrap();
        let decompressed = deflate_decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_decompress_python_zlib() {
        // Data compressed with Python: zlib.compress(bytes(range(10)), 6)
        // python3 -c "import zlib; print(list(zlib.compress(bytes(range(10)), 6)))"
        // = [120, 156, 99, 96, 100, 98, 102, 97, 101, 99, 231, 224, 4, 0, 1, 123, 0, 170]
        let compressed: Vec<u8> = vec![
            120, 156, 99, 96, 100, 98, 102, 97, 101, 99, 231, 224, 4, 0, 0, 175, 0, 46,
        ];
        let decompressed = deflate_decompress(&compressed).unwrap();
        assert_eq!(decompressed, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_compress_verifiable() {
        // Compress data and verify it decompresses correctly
        let data = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let compressed = deflate_compress(&data, 6).unwrap();
        assert!(compressed.len() > 0);
        let decompressed = deflate_decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    // --- Shuffle tests ---

    #[test]
    fn shuffle_roundtrip_f64() {
        // 4 f64 values = 32 bytes, element_size=8
        let data: Vec<u8> = (0..32).collect();
        let shuffled = shuffle_compress(&data, 8).unwrap();
        let unshuffled = shuffle_decompress(&shuffled, 8).unwrap();
        assert_eq!(unshuffled, data);
    }

    #[test]
    fn shuffle_roundtrip_i32() {
        // 8 i32 values = 32 bytes, element_size=4
        let data: Vec<u8> = (0..32).collect();
        let shuffled = shuffle_compress(&data, 4).unwrap();
        let unshuffled = shuffle_decompress(&shuffled, 4).unwrap();
        assert_eq!(unshuffled, data);
    }

    #[test]
    fn shuffle_known_pattern() {
        // 2 elements of size 4: [A0 A1 A2 A3 B0 B1 B2 B3]
        // After shuffle: [A0 B0 A1 B1 A2 B2 A3 B3]
        let data = vec![0xA0, 0xA1, 0xA2, 0xA3, 0xB0, 0xB1, 0xB2, 0xB3];
        let shuffled = shuffle_compress(&data, 4).unwrap();
        assert_eq!(shuffled, vec![0xA0, 0xB0, 0xA1, 0xB1, 0xA2, 0xB2, 0xA3, 0xB3]);
    }

    // --- Fletcher32 tests ---

    #[test]
    fn fletcher32_roundtrip() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let with_checksum = fletcher32_append(&data).unwrap();
        assert_eq!(with_checksum.len(), data.len() + 4);
        let verified = fletcher32_verify(&with_checksum).unwrap();
        assert_eq!(verified, data);
    }

    #[test]
    fn fletcher32_known_checksum() {
        // Verify checksum is deterministic
        let data = vec![0u8; 16];
        let with_checksum = fletcher32_append(&data).unwrap();
        let checksum = u32::from_le_bytes([
            with_checksum[16],
            with_checksum[17],
            with_checksum[18],
            with_checksum[19],
        ]);
        // All zeros -> sum1=0, sum2=0 -> checksum=0
        assert_eq!(checksum, 0);

        // Non-zero data
        let data2 = vec![1u8, 0, 0, 0];
        let with_checksum2 = fletcher32_append(&data2).unwrap();
        let verified = fletcher32_verify(&with_checksum2).unwrap();
        assert_eq!(verified, data2);
    }

    #[test]
    fn fletcher32_mismatch_detected() {
        let data = vec![1u8, 2, 3, 4];
        let mut with_checksum = fletcher32_append(&data).unwrap();
        // Corrupt checksum
        let last = with_checksum.len() - 1;
        with_checksum[last] ^= 0xFF;
        let result = fletcher32_verify(&with_checksum);
        assert!(matches!(result, Err(FormatError::Fletcher32Mismatch { .. })));
    }

    // --- Pipeline tests ---

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_deflate_only() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![FilterDescription {
                filter_id: FILTER_DEFLATE,
                name: None,
                flags: 0,
                client_data: vec![6],
            }],
        };
        let data: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        let compressed = compress_chunk(&data, &pipeline, 1).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, data.len(), 1).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_shuffle_deflate() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE,
                    name: None,
                    flags: 0,
                    client_data: vec![6],
                },
            ],
        };
        // 25 f64 values (200 bytes)
        let data: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        let compressed = compress_chunk(&data, &pipeline, 8).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, data.len(), 8).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_compress_decompress_roundtrip() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE,
                    name: None,
                    flags: 0,
                    client_data: vec![6],
                },
                FilterDescription {
                    filter_id: FILTER_FLETCHER32,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
            ],
        };
        let data: Vec<u8> = (0..160).map(|i| (i % 256) as u8).collect();
        let compressed = compress_chunk(&data, &pipeline, 8).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, data.len(), 8).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_shuffle_deflate_fletcher32() {
        let pipeline = FilterPipeline {
            version: 1,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE,
                    name: None,
                    flags: 0,
                    client_data: vec![9],
                },
                FilterDescription {
                    filter_id: FILTER_FLETCHER32,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
            ],
        };
        // Use realistic f64-sized data
        let data: Vec<u8> = (0..80).map(|i| (i * 3 % 256) as u8).collect();
        let compressed = compress_chunk(&data, &pipeline, 8).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, data.len(), 8).unwrap();
        assert_eq!(decompressed, data);
    }
}
