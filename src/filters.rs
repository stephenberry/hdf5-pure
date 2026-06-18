//! HDF5 filter implementations: deflate, shuffle, fletcher32.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
// `format!` is only reached by the zfp-gated code paths below.
#[cfg(all(not(feature = "std"), feature = "zfp"))]
use alloc::format;

#[cfg(feature = "zfp")]
use crate::convert::TryToUsize;
use crate::error::FormatError;
#[cfg(feature = "zfp")]
use crate::filter_pipeline::FILTER_ZFP;
use crate::filter_pipeline::{
    FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_SCALEOFFSET, FILTER_SHUFFLE, FilterPipeline,
};
use crate::scaleoffset::ScaleOffsetType;
#[cfg(feature = "zfp")]
use crate::zfp::ZfpElementType;

/// Context shared with filter pipeline operations.
///
/// Most filters (deflate, shuffle, fletcher32) need only element_size; ZFP
/// also needs chunk dimensions and scalar type, carried here so future
/// type-aware filters can look them up without changing the pipeline API.
#[derive(Debug, Clone, Copy)]
pub struct ChunkContext<'a> {
    /// Chunk dimensions in elements (one per dataset rank).
    pub chunk_dims: &'a [u64],
    /// Size of one element in bytes (for shuffle's interleave width).
    pub element_size: u32,
    /// Scalar type, required for type-aware filters like ZFP. `None` means
    /// the caller does not know or does not need it; type-aware filters
    /// will return an error.
    pub element_type: Option<ZfpElementTypeWhenEnabled>,
    /// Datatype facts the scale-offset encoder needs (class/sign/order).
    /// `None` for callers that don't have a `Datatype` or whose type isn't a
    /// scale-offset-compatible scalar; scale-offset writes then error.
    pub scale_offset_type: Option<ScaleOffsetType>,
}

/// Dummy wrapper so ChunkContext's type stays stable whether or not the
/// `zfp` feature is on. With `zfp` on this aliases `zfp::ZfpElementType`.
#[cfg(feature = "zfp")]
pub type ZfpElementTypeWhenEnabled = ZfpElementType;
#[cfg(not(feature = "zfp"))]
pub type ZfpElementTypeWhenEnabled = core::convert::Infallible;

impl<'a> ChunkContext<'a> {
    /// Lightweight constructor for callers that don't need ZFP support — the
    /// element_type is left `None`, so any ZFP filter in the pipeline will
    /// error out. `element_size` must still be valid.
    ///
    /// Currently only used by tests (read/write paths build the context via
    /// [`ChunkContext::from_datatype`]); gated so it is not shipped as dead code.
    #[cfg(test)]
    pub fn basic(chunk_dims: &'a [u64], element_size: u32) -> Self {
        Self {
            chunk_dims,
            element_size,
            element_type: None,
            scale_offset_type: None,
        }
    }

    /// Build a full context from a dataset's `Datatype`: derives
    /// `element_size` from `dt.type_size()` and `element_type` from
    /// [`zfp_element_type_from_datatype`]. This is the preferred
    /// constructor for read/write paths where a `Datatype` is in scope,
    /// so the two fields can't drift out of sync.
    pub fn from_datatype(chunk_dims: &'a [u64], dt: &crate::datatype::Datatype) -> Self {
        Self {
            chunk_dims,
            element_size: dt.type_size(),
            element_type: zfp_element_type_from_datatype(dt),
            scale_offset_type: crate::scaleoffset::scale_offset_type_from_datatype(dt),
        }
    }
}

/// Map an HDF5 `Datatype` to the matching ZFP scalar type, if it's one of the
/// supported codec widths. Returns `None` for types outside f32/f64/i32/i64.
#[cfg(feature = "zfp")]
pub fn zfp_element_type_from_datatype(
    dt: &crate::datatype::Datatype,
) -> Option<ZfpElementTypeWhenEnabled> {
    use crate::datatype::Datatype;
    match dt {
        Datatype::FloatingPoint { size: 4, .. } => Some(ZfpElementType::F32),
        Datatype::FloatingPoint { size: 8, .. } => Some(ZfpElementType::F64),
        Datatype::FixedPoint {
            size: 4,
            signed: true,
            ..
        } => Some(ZfpElementType::I32),
        Datatype::FixedPoint {
            size: 8,
            signed: true,
            ..
        } => Some(ZfpElementType::I64),
        _ => None,
    }
}

#[cfg(not(feature = "zfp"))]
pub fn zfp_element_type_from_datatype(
    _: &crate::datatype::Datatype,
) -> Option<ZfpElementTypeWhenEnabled> {
    None
}

/// Apply a filter pipeline to decompress a chunk.
/// Filters are applied in REVERSE order for decompression.
pub fn decompress_chunk(
    compressed: &[u8],
    pipeline: &FilterPipeline,
    ctx: ChunkContext<'_>,
    filter_mask: u32,
) -> Result<Vec<u8>, FormatError> {
    // Expected size of the fully decoded chunk. Every chunk, even one straddling
    // a dataset edge, is stored at full chunk size, so this is the exact decoded
    // length. Used to bound deflate output (decompression-bomb guard) and to
    // reject a chunk that decodes to the wrong size.
    let expected = expected_chunk_len(&ctx);

    let mut owned: Option<Vec<u8>> = None;
    // Filters are listed in application (forward) order; decoding reverses them.
    // `i` is the filter's forward index, which is also its bit position in
    // `filter_mask` (HDF5 H5Z pipeline numbering): bit `i` set means filter `i`
    // was skipped for THIS chunk and must NOT be reversed. Treating any non-zero
    // mask as "return raw" (the prior behaviour) corrupts chunks in a multi-filter
    // pipeline where only some filters were skipped (e.g. shuffle+gzip on an
    // incompressible chunk, which is stored shuffled but not deflated).
    for (i, filter) in pipeline.filters.iter().enumerate().rev() {
        if i < 32 && (filter_mask >> i) & 1 == 1 {
            continue;
        }
        let input: &[u8] = owned.as_deref().unwrap_or(compressed);
        let next = match filter.filter_id {
            FILTER_SHUFFLE => shuffle_decompress(input, ctx.element_size as usize)?,
            FILTER_DEFLATE => {
                deflate_decompress(input, deflate_output_cap(expected, pipeline, filter_mask, i))?
            }
            FILTER_FLETCHER32 => fletcher32_verify(input)?,
            FILTER_SCALEOFFSET => crate::scaleoffset::decompress(input, filter)?,
            #[cfg(feature = "zfp")]
            FILTER_ZFP => zfp_decompress(input, filter, &ctx)?,
            other => return Err(FormatError::UnsupportedFilter(other)),
        };
        owned = Some(next);
    }
    let result = owned.unwrap_or_else(|| compressed.to_vec());

    // A valid chunk always decodes to exactly the full chunk size. A mismatch
    // means a corrupt or hostile filter stream; erroring here prevents silently
    // zero-filling (when short) or dropping (when long) data during chunk
    // assembly, which copies only the in-range overlap.
    if let Some(expected) = expected {
        if result.len() != expected {
            return Err(FormatError::DataSizeMismatch {
                expected,
                actual: result.len(),
            });
        }
    }
    Ok(result)
}

/// Expected byte length of a fully decoded chunk: product of the chunk element
/// dimensions times the element size. Returns `None` when the product can't be
/// represented (treated as "unknown", so the size-dependent guards are skipped
/// rather than misfiring) or is zero.
fn expected_chunk_len(ctx: &ChunkContext<'_>) -> Option<usize> {
    let elems = ctx
        .chunk_dims
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))?;
    let bytes = elems.checked_mul(u64::from(ctx.element_size))?;
    usize::try_from(bytes).ok().filter(|&n| n != 0)
}

/// Upper bound for a deflate stage's decoded output. On decode, deflate is
/// reversed BEFORE the lower-forward-index filters that ran before it when the
/// chunk was written; any of those that EXPAND the data make deflate's
/// legitimate output larger than the final chunk size. Among supported filters
/// only Fletcher32 expands (by its 4-byte trailing checksum, reversed after
/// deflate when it sits at a lower forward index), so the cap is the final chunk
/// size plus 4 per surviving inner Fletcher32. `None` (size unknown) stays
/// uncapped. The exact-size check after the whole pipeline still rejects
/// genuinely wrong output, so this only needs to bound memory (the
/// decompression-bomb guard) without rejecting a valid chunk.
fn deflate_output_cap(
    expected: Option<usize>,
    pipeline: &FilterPipeline,
    filter_mask: u32,
    deflate_index: usize,
) -> Option<usize> {
    let expected = expected?;
    let inner_overhead: usize = pipeline.filters[..deflate_index]
        .iter()
        .enumerate()
        .filter(|(j, _)| !(*j < 32 && (filter_mask >> *j) & 1 == 1))
        .map(|(_, f)| if f.filter_id == FILTER_FLETCHER32 { 4 } else { 0 })
        .sum();
    Some(expected.saturating_add(inner_overhead))
}

/// Apply a filter pipeline to compress a chunk.
/// Filters are applied in FORWARD order for compression.
pub fn compress_chunk(
    data: &[u8],
    pipeline: &FilterPipeline,
    ctx: ChunkContext<'_>,
) -> Result<Vec<u8>, FormatError> {
    let mut owned: Option<Vec<u8>> = None;
    for filter in &pipeline.filters {
        let input: &[u8] = owned.as_deref().unwrap_or(data);
        let next = match filter.filter_id {
            FILTER_SHUFFLE => shuffle_compress(input, ctx.element_size as usize)?,
            FILTER_DEFLATE => {
                let level = filter.client_data.first().copied().unwrap_or(6);
                deflate_compress(input, level)?
            }
            FILTER_FLETCHER32 => fletcher32_append(input)?,
            FILTER_SCALEOFFSET => crate::scaleoffset::compress(input, filter)?,
            #[cfg(feature = "zfp")]
            FILTER_ZFP => zfp_compress(input, filter, &ctx)?,
            other => return Err(FormatError::UnsupportedFilter(other)),
        };
        owned = Some(next);
    }
    Ok(owned.unwrap_or_else(|| data.to_vec()))
}

#[cfg(feature = "zfp")]
fn zfp_rate(filter: &crate::filter_pipeline::FilterDescription) -> Result<f64, FormatError> {
    crate::zfp::zfp_rate_from_cd_values(&filter.client_data)
        .ok_or_else(|| FormatError::FilterError("ZFP: invalid or non-rate cd_values".into()))
}

#[cfg(feature = "zfp")]
fn zfp_element_type(ctx: &ChunkContext<'_>) -> Result<ZfpElementType, FormatError> {
    ctx.element_type.ok_or_else(|| {
        FormatError::FilterError(
            "ZFP: element_type missing from ChunkContext (caller must set it)".into(),
        )
    })
}

/// Copy chunk dims into a stack buffer and return a slice of the valid
/// prefix. ZFP's rank bound is 4, so a heap Vec is unnecessary per chunk.
#[cfg(feature = "zfp")]
fn zfp_dims_on_stack(ctx: &ChunkContext<'_>) -> Result<([usize; 4], usize), FormatError> {
    let rank = ctx.chunk_dims.len();
    if rank == 0 || rank > 4 {
        return Err(FormatError::FilterError(format!(
            "ZFP: chunk rank must be 1..=4, got {rank}",
        )));
    }
    let mut buf = [0usize; 4];
    for (slot, &d) in buf.iter_mut().zip(ctx.chunk_dims.iter()) {
        *slot = d.to_usize()?;
    }
    Ok((buf, rank))
}

#[cfg(feature = "zfp")]
fn zfp_compress(
    data: &[u8],
    filter: &crate::filter_pipeline::FilterDescription,
    ctx: &ChunkContext<'_>,
) -> Result<Vec<u8>, FormatError> {
    let rate = zfp_rate(filter)?;
    let elem_ty = zfp_element_type(ctx)?;
    let (dims_buf, rank) = zfp_dims_on_stack(ctx)?;
    crate::zfp::compress(data, &dims_buf[..rank], rate, elem_ty)
}

#[cfg(feature = "zfp")]
fn zfp_decompress(
    data: &[u8],
    filter: &crate::filter_pipeline::FilterDescription,
    ctx: &ChunkContext<'_>,
) -> Result<Vec<u8>, FormatError> {
    let rate = zfp_rate(filter)?;
    let elem_ty = zfp_element_type(ctx)?;
    let (dims_buf, rank) = zfp_dims_on_stack(ctx)?;
    crate::zfp::decompress(data, &dims_buf[..rank], rate, elem_ty)
}

/// Decompress zlib-compressed data.
///
/// `max_output`, when known, bounds the decompressed size: a deflate stage in a
/// chunk pipeline never expands beyond the chunk's expected byte size, so a
/// stream that inflates past it signals a decompression bomb and is rejected
/// instead of being allowed to allocate unbounded memory (OOM).
#[cfg(feature = "deflate")]
fn deflate_decompress(data: &[u8], max_output: Option<usize>) -> Result<Vec<u8>, FormatError> {
    use std::io::Read;
    let decoder = flate2::read::ZlibDecoder::new(data);
    let mut result = Vec::new();
    match max_output {
        Some(limit) => {
            // Read at most `limit + 1` bytes: anything beyond `limit` proves the
            // stream exceeds the expected chunk size, so reject rather than OOM.
            let cap = (limit as u64).saturating_add(1);
            decoder
                .take(cap)
                .read_to_end(&mut result)
                .map_err(|e| FormatError::DecompressionError(e.to_string()))?;
            if result.len() > limit {
                return Err(FormatError::DecompressionError(format!(
                    "deflate output exceeds expected chunk size of {limit} bytes \
                     (possible decompression bomb)"
                )));
            }
        }
        None => {
            let mut decoder = decoder;
            decoder
                .read_to_end(&mut result)
                .map_err(|e| FormatError::DecompressionError(e.to_string()))?;
        }
    }
    Ok(result)
}

#[cfg(not(feature = "deflate"))]
fn deflate_decompress(_data: &[u8], _max_output: Option<usize>) -> Result<Vec<u8>, FormatError> {
    Err(FormatError::UnsupportedFilter(FILTER_DEFLATE))
}

/// Compress data with zlib.
#[cfg(feature = "deflate")]
fn deflate_compress(data: &[u8], level: u32) -> Result<Vec<u8>, FormatError> {
    use std::io::Write;
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(level));
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
        let decompressed = deflate_decompress(&compressed, None).unwrap();
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
        let decompressed = deflate_decompress(&compressed, None).unwrap();
        assert_eq!(decompressed, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_compress_verifiable() {
        // Compress data and verify it decompresses correctly
        let data = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let compressed = deflate_compress(&data, 6).unwrap();
        assert!(!compressed.is_empty());
        let decompressed = deflate_decompress(&compressed, None).unwrap();
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
        assert_eq!(
            shuffled,
            vec![0xA0, 0xB0, 0xA1, 0xB1, 0xA2, 0xB2, 0xA3, 0xB3]
        );
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
        assert!(matches!(
            result,
            Err(FormatError::Fletcher32Mismatch { .. })
        ));
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
        let dims = [data.len() as u64];
        let ctx = ChunkContext::basic(&dims, 1);
        let compressed = compress_chunk(&data, &pipeline, ctx).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, ctx, 0).unwrap();
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
        let dims = [(data.len() / 8) as u64];
        let ctx = ChunkContext::basic(&dims, 8);
        let compressed = compress_chunk(&data, &pipeline, ctx).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, ctx, 0).unwrap();
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
        let dims = [(data.len() / 8) as u64];
        let ctx = ChunkContext::basic(&dims, 8);
        let compressed = compress_chunk(&data, &pipeline, ctx).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, ctx, 0).unwrap();
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
        let dims = [(data.len() / 8) as u64];
        let ctx = ChunkContext::basic(&dims, 8);
        let compressed = compress_chunk(&data, &pipeline, ctx).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, ctx, 0).unwrap();
        assert_eq!(decompressed, data);
    }

    /// A non-zero `filter_mask` is per-filter: only the masked filters were
    /// skipped for this chunk; the rest still apply. The common case is a
    /// shuffle+deflate pipeline where an incompressible chunk is stored shuffled
    /// but NOT deflated. Decoding must reverse shuffle while skipping deflate.
    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_partial_mask_reverses_surviving_filter() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE, // forward index 0
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE, // forward index 1
                    name: None,
                    flags: 0,
                    client_data: vec![6],
                },
            ],
        };
        let data: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        let dims = [(data.len() / 8) as u64];
        let ctx = ChunkContext::basic(&dims, 8);

        // Stored form when deflate was declined: shuffled only.
        let stored = shuffle_compress(&data, 8).unwrap();
        // Bit 1 set => deflate (index 1) was skipped for this chunk.
        let mask = 1u32 << 1;
        let decoded = decompress_chunk(&stored, &pipeline, ctx, mask).unwrap();
        assert_eq!(decoded, data, "shuffle must be reversed even when deflate is skipped");

        // The previous behaviour returned raw (still-shuffled) bytes — guard it.
        assert_ne!(stored, data, "precondition: stored bytes are shuffled, not raw");
    }

    /// Symmetric case: the low filter is skipped, the high one still applies.
    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_partial_mask_skips_low_filter() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE, // forward index 0
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE, // forward index 1
                    name: None,
                    flags: 0,
                    client_data: vec![6],
                },
            ],
        };
        let data: Vec<u8> = (0u32..200).map(|i| (i.wrapping_mul(7) % 256) as u8).collect();
        let dims = [(data.len() / 8) as u64];
        let ctx = ChunkContext::basic(&dims, 8);

        // Shuffle skipped: stored = deflate(data) directly.
        let stored = deflate_compress(&data, 6).unwrap();
        let mask = 1u32 << 0; // bit 0 => shuffle (index 0) skipped
        let decoded = decompress_chunk(&stored, &pipeline, ctx, mask).unwrap();
        assert_eq!(decoded, data);
    }

    // --- Decompression-bomb / size guards (#5) ---

    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_decompress_rejects_bomb() {
        // A few bytes that inflate to 100 KB; with a 1 KB cap this is rejected
        // rather than allowed to allocate unbounded memory.
        let huge = vec![0u8; 100_000];
        let compressed = deflate_compress(&huge, 9).unwrap();
        assert!(compressed.len() < 1024);
        let err = deflate_decompress(&compressed, Some(1024)).unwrap_err();
        assert!(matches!(err, FormatError::DecompressionError(_)));
        // Without a cap it still works (used where the size is genuinely unknown).
        assert_eq!(deflate_decompress(&compressed, None).unwrap().len(), 100_000);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_decompress_within_cap_ok() {
        let data = vec![7u8; 500];
        let compressed = deflate_compress(&data, 6).unwrap();
        // Cap equal to the exact output length must pass.
        assert_eq!(deflate_decompress(&compressed, Some(500)).unwrap(), data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn decompress_chunk_rejects_wrong_decoded_size() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![FilterDescription {
                filter_id: FILTER_DEFLATE,
                name: None,
                flags: 0,
                client_data: vec![6],
            }],
        };
        // Chunk decodes to 50 bytes, but the context expects 100 (10 elems x 10).
        let data = vec![3u8; 50];
        let compressed = compress_chunk(&data, &pipeline, ChunkContext::basic(&[50], 1)).unwrap();
        let ctx = ChunkContext::basic(&[10], 10); // expected = 100 bytes
        let err = decompress_chunk(&compressed, &pipeline, ctx, 0).unwrap_err();
        assert!(matches!(err, FormatError::DataSizeMismatch { expected: 100, actual: 50 }));
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_fletcher32_inner_deflate_outer_roundtrips() {
        // Fletcher32 BEFORE deflate on the write path (forward index 0): the
        // 4-byte checksum is appended first, then deflate compresses data+4. On
        // decode, deflate is reversed first and legitimately produces
        // `expected + 4` bytes, which must NOT be mistaken for a decompression
        // bomb by the deflate output cap.
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_FLETCHER32, // forward index 0 (inner)
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE, // forward index 1 (outer)
                    name: None,
                    flags: 0,
                    client_data: vec![6],
                },
            ],
        };
        let data: Vec<u8> = (0u32..200).map(|i| (i % 256) as u8).collect();
        let ctx = ChunkContext::basic(&[200], 1); // expected = 200
        let compressed = compress_chunk(&data, &pipeline, ctx).unwrap();
        let decoded = decompress_chunk(&compressed, &pipeline, ctx, 0).unwrap();
        assert_eq!(decoded, data);
    }
}
