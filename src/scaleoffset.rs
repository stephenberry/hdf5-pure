//! HDF5 Scale-Offset filter (filter id 6).
//!
//! Two modes, matching the only two the reference HDF5 library implements:
//!
//! * **Integer** — lossless. Each chunk's minimum is subtracted and the
//!   residuals are packed into the fewest bits that cover the chunk's range.
//! * **Float D-scale** — lossy. Values are multiplied by `10^decimals`,
//!   rounded to integers, then compressed like the integer mode.
//!
//! Ported faithfully from the reference `H5Zscaleoffset.c` so files we write
//! are read by the C library and vice versa. The on-disk compressed chunk is
//! a fixed 21-byte header followed by an MSB-first bitstream of per-element
//! offsets:
//!
//! | bytes   | meaning                                  |
//! |---------|------------------------------------------|
//! | `0..4`  | `minbits` (u32, little-endian)           |
//! | `4`     | size of the `minval` field, always `8`   |
//! | `5..13` | `minval` (u64, little-endian)            |
//! | `13..21`| zero padding                             |
//! | `21..`  | payload: each offset in `minbits` bits   |
//!
//! The payload bitstream is endianness-independent (it encodes the integer
//! offset MSB-first); the dataset byte order only governs how reconstructed
//! values are serialized.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{string::ToString, vec, vec::Vec};

use crate::convert::TryToUsize;
use crate::datatype::{Datatype, DatatypeByteOrder};
use crate::error::FormatError;
use crate::filter_pipeline::FilterDescription;

// cd_values indices (H5Z_SCALEOFFSET_PARM_*).
const PARM_SCALETYPE: usize = 0;
const PARM_SCALEFACTOR: usize = 1;
const PARM_NELMTS: usize = 2;
const PARM_CLASS: usize = 3;
const PARM_SIZE: usize = 4;
const PARM_SIGN: usize = 5;
const PARM_ORDER: usize = 6;
const PARM_FILAVAIL: usize = 7;
const PARM_FILVAL: usize = 8;
/// Total number of filter parameters (`H5Z_SCALEOFFSET_TOTAL_NPARMS`).
const TOTAL_NPARMS: usize = 20;
/// The first `PARM_FILVAL` entries are the always-present "core" parameters.
const CORE_NPARMS: usize = PARM_FILVAL;

// Scale types (`H5Z_SO_scale_type_t`).
const SO_FLOAT_DSCALE: u32 = 0;
const SO_FLOAT_ESCALE: u32 = 1;
const SO_INT: u32 = 2;

// Datatype classes.
const CLS_INTEGER: u32 = 0;
const CLS_FLOAT: u32 = 1;

// Integer sign.
const SGN_NONE: u32 = 0;
const SGN_2: u32 = 1;

// Byte order.
const ORDER_LE: u32 = 0;
const ORDER_BE: u32 = 1;

// Fill-value availability.
const FILL_UNDEFINED: u32 = 0;
const FILL_DEFINED: u32 = 1;

/// Length of the fixed parameter header that precedes the bit-packed payload
/// (`buf_offset` in the reference filter).
const HEADER_LEN: usize = 21;

/// Scale-offset compression mode requested by the writer.
///
/// Mirrors the two variants the reference HDF5 library exposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleOffset {
    /// Integer scale-offset (lossless). `0` lets the encoder auto-compute the
    /// minimum bit width from each chunk's value range (the usual choice); a
    /// positive value would force a fixed minimum bit width.
    Integer(u32),
    /// Floating-point decimal scaling (lossy). The value is the number of
    /// decimal digits of precision retained (the "D" scale factor).
    FloatDScale(i32),
}

/// Datatype facts the writer needs to assemble scale-offset `cd_values`.
///
/// Derived from a dataset's [`Datatype`] via
/// [`scale_offset_type_from_datatype`]. Carried on
/// [`ChunkContext`](crate::filters::ChunkContext) the same way ZFP's scalar
/// type is, so the write path can build the filter parameters without
/// re-deriving them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaleOffsetType {
    /// `CLS_INTEGER` or `CLS_FLOAT`.
    class: u32,
    /// `SGN_NONE` or `SGN_2` (only meaningful for integers).
    sign: u32,
    /// `ORDER_LE` or `ORDER_BE`.
    order: u32,
}

/// Map an HDF5 `Datatype` to the facts scale-offset needs, if the type is a
/// fixed-point or floating-point scalar with a definite byte order. Returns
/// `None` for any other class (compound, string, …) or an indeterminate order.
pub fn scale_offset_type_from_datatype(dt: &Datatype) -> Option<ScaleOffsetType> {
    match dt {
        Datatype::FixedPoint {
            size,
            byte_order,
            signed,
            ..
        } if matches!(*size, 1 | 2 | 4 | 8) => Some(ScaleOffsetType {
            class: CLS_INTEGER,
            sign: if *signed { SGN_2 } else { SGN_NONE },
            order: order_code(byte_order)?,
        }),
        Datatype::FloatingPoint {
            size, byte_order, ..
        } if matches!(*size, 4 | 8) => Some(ScaleOffsetType {
            class: CLS_FLOAT,
            sign: SGN_NONE,
            order: order_code(byte_order)?,
        }),
        _ => None,
    }
}

fn order_code(order: &DatatypeByteOrder) -> Option<u32> {
    match order {
        DatatypeByteOrder::LittleEndian => Some(ORDER_LE),
        DatatypeByteOrder::BigEndian => Some(ORDER_BE),
        // Scale-offset only supports definite little/big endian, matching the
        // reference filter's `can_apply` check.
        DatatypeByteOrder::Vax => None,
    }
}

/// Build the 20-entry `cd_values` array for a scale-offset filter.
///
/// `nelmts` is the number of elements in one chunk. Validates that the
/// requested [`ScaleOffset`] mode matches the datatype class (integer mode on
/// integer data, float D-scale on float data).
pub fn build_cd_values(
    mode: ScaleOffset,
    ty: ScaleOffsetType,
    size: u32,
    nelmts: u32,
) -> Result<Vec<u32>, FormatError> {
    let (scale_type, scale_factor) = match (mode, ty.class) {
        (ScaleOffset::Integer(minbits), CLS_INTEGER) => (SO_INT, minbits),
        (ScaleOffset::FloatDScale(decimals), CLS_FLOAT) => (SO_FLOAT_DSCALE, decimals as u32),
        (ScaleOffset::Integer(_), _) => {
            return Err(FormatError::FilterError(
                "scaleoffset: integer mode requires an integer dataset".to_string(),
            ));
        }
        (ScaleOffset::FloatDScale(_), _) => {
            return Err(FormatError::FilterError(
                "scaleoffset: float D-scale mode requires a floating-point dataset".to_string(),
            ));
        }
    };

    let mut cd = vec![0u32; TOTAL_NPARMS];
    cd[PARM_SCALETYPE] = scale_type;
    cd[PARM_SCALEFACTOR] = scale_factor;
    cd[PARM_NELMTS] = nelmts;
    cd[PARM_CLASS] = ty.class;
    cd[PARM_SIZE] = size;
    cd[PARM_SIGN] = ty.sign;
    cd[PARM_ORDER] = ty.order;
    cd[PARM_FILAVAIL] = FILL_UNDEFINED;
    Ok(cd)
}

/// Decoded scale-offset parameters shared by the compress and decompress paths.
struct Parms {
    scale_type: u32,
    scale_factor: i32,
    nelmts: usize,
    class: u32,
    size: usize,
    order: u32,
    filavail: u32,
}

impl Parms {
    fn parse(cd: &[u32]) -> Result<Parms, FormatError> {
        if cd.len() < CORE_NPARMS {
            return Err(FormatError::FilterError(
                "scaleoffset: too few cd_values".to_string(),
            ));
        }
        let class = cd[PARM_CLASS];
        if class != CLS_INTEGER && class != CLS_FLOAT {
            return Err(FormatError::FilterError(
                "scaleoffset: unsupported datatype class".to_string(),
            ));
        }
        let size = cd[PARM_SIZE] as usize;
        if size == 0 || size > 8 {
            return Err(FormatError::FilterError(
                "scaleoffset: unsupported datatype size".to_string(),
            ));
        }
        if class == CLS_FLOAT && size != 4 && size != 8 {
            return Err(FormatError::FilterError(
                "scaleoffset: float size must be 4 or 8".to_string(),
            ));
        }
        let order = cd[PARM_ORDER];
        if order != ORDER_LE && order != ORDER_BE {
            return Err(FormatError::FilterError(
                "scaleoffset: bad byte order".to_string(),
            ));
        }
        Ok(Parms {
            scale_type: cd[PARM_SCALETYPE],
            scale_factor: cd[PARM_SCALEFACTOR] as i32,
            nelmts: cd[PARM_NELMTS].to_usize()?,
            class,
            size,
            order,
            filavail: cd[PARM_FILAVAIL],
        })
    }

    /// Bit mask covering the datatype's full width.
    fn width_mask(&self) -> u64 {
        if self.size >= 8 {
            u64::MAX
        } else {
            (1u64 << (self.size * 8)) - 1
        }
    }
}

/// Decompress one scale-offset chunk into raw element bytes (in the dataset's
/// stored byte order).
pub fn decompress(input: &[u8], filter: &FilterDescription) -> Result<Vec<u8>, FormatError> {
    let cd = &filter.client_data;
    let p = Parms::parse(cd)?;

    if p.scale_type == SO_FLOAT_ESCALE {
        return Err(FormatError::FilterError(
            "scaleoffset: float E-scale method is not supported".to_string(),
        ));
    }

    let full_bits = (p.size * 8) as u32;
    // `nelmts` is file-derived; `nelmts * size` is an allocation size and a
    // slice bound. The product is computed in `u64` (it cannot overflow there:
    // `nelmts` originates from a `u32` and `size` is 1..=8) and narrowed to
    // `usize`, which errors instead of truncating on a 32-bit host.
    let size_out = (p.nelmts as u64 * p.size as u64).to_usize()?;

    // No-op mode: integer (non-DSCALE) datasets created with scale_factor equal
    // to the full bit width store the chunk verbatim, with no header.
    if p.scale_type != SO_FLOAT_DSCALE && p.scale_factor == full_bits as i32 {
        return Ok(input.to_vec());
    }

    // Header: minbits + minval.
    if input.len() < 5 {
        return Err(FormatError::FilterError(
            "scaleoffset: chunk shorter than header".to_string(),
        ));
    }
    let minbits = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);
    if minbits > full_bits {
        return Err(FormatError::FilterError(
            "scaleoffset: minbits exceeds datatype size".to_string(),
        ));
    }
    let minval_size = (input[4] as usize).min(8);
    if input.len() < 5 + minval_size {
        return Err(FormatError::FilterError(
            "scaleoffset: chunk too short for minval".to_string(),
        ));
    }
    let mut minval_bytes = [0u8; 8];
    minval_bytes[..minval_size].copy_from_slice(&input[5..5 + minval_size]);
    let minval = u64::from_le_bytes(minval_bytes);

    // Raw payload (no per-element packing): minbits at full precision.
    if minbits == full_bits {
        let start = HEADER_LEN;
        if input.len() < start + size_out {
            return Err(FormatError::FilterError(
                "scaleoffset: chunk too short for raw payload".to_string(),
            ));
        }
        return Ok(input[start..start + size_out].to_vec());
    }

    // minbits == 0 means every element equals `minval`. Short-circuit so we
    // never enter the sentinel-matching path, which would otherwise misread
    // a `FILL_DEFINED` chunk's stored values as fill values (the all-ones
    // sentinel collapses to 0 when minbits == 0, the same value every zero
    // offset carries). The reference C decoder handles this identically.
    let mut out = Vec::with_capacity(size_out);
    if minbits == 0 {
        let mask = p.width_mask();
        for _ in 0..p.nelmts {
            write_value(&mut out, minval & mask, p.size, p.order);
        }
        return Ok(out);
    }

    if input.len() < HEADER_LEN {
        return Err(FormatError::FilterError(
            "scaleoffset: chunk too short for packed payload".to_string(),
        ));
    }
    let payload = &input[HEADER_LEN..];
    // Validate payload length once, up front; the BitReader trusts the
    // caller to not read past the end. Compare in `u64` so the bit counts
    // (`nelmts` is file-derived, `minbits` up to 64) cannot overflow a 32-bit
    // `usize` and falsely pass the check.
    if (payload.len() as u64) * 8 < p.nelmts as u64 * minbits as u64 {
        return Err(FormatError::FilterError(
            "scaleoffset: payload too short for packed data".to_string(),
        ));
    }
    let mut reader = BitReader::new(payload);

    if p.class == CLS_INTEGER {
        reconstruct_integer(&mut out, &mut reader, &p, minbits, minval, cd)?;
    } else {
        reconstruct_float(&mut out, &mut reader, &p, minbits, minval, cd)?;
    }
    Ok(out)
}

/// Compress one full chunk of raw element bytes with scale-offset.
///
/// `input` must be exactly `nelmts * size` bytes (the chunk writer always pads
/// edge chunks to full size). Only the fill-value-undefined path is produced,
/// which is what this crate's writer emits.
pub fn compress(input: &[u8], filter: &FilterDescription) -> Result<Vec<u8>, FormatError> {
    let cd = &filter.client_data;
    let p = Parms::parse(cd)?;

    if p.filavail == FILL_DEFINED {
        return Err(FormatError::FilterError(
            "scaleoffset: encoding with a defined fill value is not supported".to_string(),
        ));
    }
    if p.class == CLS_INTEGER && p.scale_type != SO_INT {
        return Err(FormatError::FilterError(
            "scaleoffset: integer class requires integer scale type".to_string(),
        ));
    }
    if p.class == CLS_FLOAT && p.scale_type != SO_FLOAT_DSCALE {
        return Err(FormatError::FilterError(
            "scaleoffset: float class requires D-scale scale type".to_string(),
        ));
    }

    // `nelmts * size` in `u64` (cannot overflow there) narrowed to `usize`,
    // so a file-derived `nelmts` cannot wrap a 32-bit `usize` and spuriously
    // equal `input.len()`.
    let expected = (p.nelmts as u64 * p.size as u64).to_usize()?;
    if input.len() != expected {
        return Err(FormatError::CompressionError(
            "scaleoffset: chunk size does not match nelmts * datatype size".to_string(),
        ));
    }
    if p.nelmts == 0 {
        return Ok(emit(0, 0, &[]));
    }

    let signed = cd[PARM_SIGN] == SGN_2;
    let full_bits = (p.size * 8) as u32;

    let (minbits, minval, offsets) = if p.class == CLS_INTEGER {
        precompress_integer(input, &p, signed)
    } else {
        precompress_float(input, &p)
    };

    // Raw path: store the original element bytes after the header.
    if minbits >= full_bits {
        return Ok(emit_raw(full_bits, minval, input));
    }
    let payload = pack_offsets(&offsets, minbits, p.nelmts)?;
    Ok(emit(minbits, minval, &payload))
}

// --- integer reconstruction / pre-compression ----------------------------

fn reconstruct_integer(
    out: &mut Vec<u8>,
    reader: &mut BitReader<'_>,
    p: &Parms,
    minbits: u32,
    minval: u64,
    cd: &[u32],
) -> Result<(), FormatError> {
    let mask = p.width_mask();
    let sentinel = sentinel(minbits);
    // Hoist `fv & mask` out of the per-element branch.
    let filval = if p.filavail == FILL_DEFINED {
        Some(read_fill_bits(cd, p.size)? & mask)
    } else {
        None
    };
    for _ in 0..p.nelmts {
        let d = reader.read(minbits);
        let bits = match filval {
            Some(fv) if d == sentinel => fv,
            _ => d.wrapping_add(minval) & mask,
        };
        write_value(out, bits, p.size, p.order);
    }
    Ok(())
}

fn precompress_integer(input: &[u8], p: &Parms, signed: bool) -> (u32, u64, Vec<u64>) {
    // Stream the input twice (min/max, then offsets) instead of materializing
    // a Vec<i128>. The intermediate dominated cache pressure on large chunks
    // (16 B/element vs the source's 1-8 B/element); two cache-warm passes
    // come out ahead.
    let full_bits = (p.size * 8) as u32;
    let read_as_i128 = |i: usize| -> i128 {
        let bits = read_value(&input[i * p.size..(i + 1) * p.size], p.size, p.order);
        if signed {
            sign_extend(bits, p.size) as i128
        } else {
            bits as i128
        }
    };

    let first = read_as_i128(0);
    let (mut min, mut max) = (first, first);
    for i in 1..p.nelmts {
        let v = read_as_i128(i);
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }

    let minval = if signed {
        (min as i64) as u64
    } else {
        min as u64
    };

    // Overflow guard mirrors `H5Z_scaleoffset_check_{1,2}`: a span within two
    // of the full range can't gain from packing, so store at full precision.
    let width_max: u128 = p.width_mask() as u128;
    let spread = (max - min) as u128;
    if spread > width_max.saturating_sub(2) {
        return (full_bits, minval, Vec::new());
    }

    let minbits = ceil_log2((spread as u64) + 1);
    if minbits >= full_bits {
        return (full_bits, minval, Vec::new());
    }

    let offsets = (0..p.nelmts)
        .map(|i| (read_as_i128(i) - min) as u64)
        .collect();
    (minbits, minval, offsets)
}

// --- float (D-scale) reconstruction / pre-compression ---------------------

fn reconstruct_float(
    out: &mut Vec<u8>,
    reader: &mut BitReader<'_>,
    p: &Parms,
    minbits: u32,
    minval: u64,
    cd: &[u32],
) -> Result<(), FormatError> {
    let sentinel = sentinel(minbits);
    let decimals = p.scale_factor;
    let filval = if p.filavail == FILL_DEFINED {
        Some(read_fill_bits(cd, p.size)?)
    } else {
        None
    };
    // Dispatch on width once: pow, min, and the size/order args are all
    // loop-invariant. The compiler won't hoist the inner `match p.size`
    // across both branches by itself.
    if p.size == 4 {
        let min = f32::from_bits(minval as u32);
        let pow = pow10_f32(decimals);
        for _ in 0..p.nelmts {
            let d = reader.read(minbits);
            let bits = if let Some(fv) = filval.filter(|_| d == sentinel) {
                fv
            } else {
                ((d as i64 as f32) / pow + min).to_bits() as u64
            };
            write_value(out, bits, 4, p.order);
        }
    } else {
        let min = f64::from_bits(minval);
        let pow = pow10_f64(decimals);
        for _ in 0..p.nelmts {
            let d = reader.read(minbits);
            let bits = if let Some(fv) = filval.filter(|_| d == sentinel) {
                fv
            } else {
                ((d as i64 as f64) / pow + min).to_bits()
            };
            write_value(out, bits, 8, p.order);
        }
    }
    Ok(())
}

fn precompress_float(input: &[u8], p: &Parms) -> (u32, u64, Vec<u64>) {
    // Two streaming passes over the input bytes (min/max, then offsets),
    // no intermediate Vec<f32>/Vec<f64>. `min_scaled = min * pow` is hoisted
    // out of the per-element loop so we don't recompute the same product N
    // times — bit-identical to the previous `v * pow - min * pow`.
    let decimals = p.scale_factor;
    let full_bits = (p.size * 8) as u32;
    if p.size == 4 {
        let read_f32 = |i: usize| -> f32 {
            f32::from_bits(read_value(&input[i * 4..i * 4 + 4], 4, p.order) as u32)
        };
        let first = read_f32(0);
        let (mut min, mut max) = (first, first);
        for i in 1..p.nelmts {
            let v = read_f32(i);
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        let pow = pow10_f32(decimals);
        let min_scaled = min * pow;
        // check_3: residual span beyond signed 32-bit range → store raw.
        let residual = max * pow - min_scaled;
        let minval = min.to_bits() as u64;
        if residual > (1u64 << 31) as f32 {
            return (full_bits, minval, Vec::new());
        }
        let minbits = ceil_log2((round_half_away_f32(residual) as u64) + 1);
        if minbits >= full_bits {
            return (full_bits, minval, Vec::new());
        }
        let offsets = (0..p.nelmts)
            .map(|i| round_half_away_f32(read_f32(i) * pow - min_scaled) as u64)
            .collect();
        (minbits, minval, offsets)
    } else {
        let read_f64 =
            |i: usize| -> f64 { f64::from_bits(read_value(&input[i * 8..i * 8 + 8], 8, p.order)) };
        let first = read_f64(0);
        let (mut min, mut max) = (first, first);
        for i in 1..p.nelmts {
            let v = read_f64(i);
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        let pow = pow10_f64(decimals);
        let min_scaled = min * pow;
        let residual = max * pow - min_scaled;
        let minval = min.to_bits();
        if residual > (1u64 << 63) as f64 {
            return (full_bits, minval, Vec::new());
        }
        let minbits = ceil_log2((round_half_away_f64(residual) as u64) + 1);
        if minbits >= full_bits {
            return (full_bits, minval, Vec::new());
        }
        let offsets = (0..p.nelmts)
            .map(|i| round_half_away_f64(read_f64(i) * pow - min_scaled) as u64)
            .collect();
        (minbits, minval, offsets)
    }
}

// --- header emit helpers --------------------------------------------------

/// Assemble a compressed chunk: 21-byte header + bit-packed `payload`.
/// The trailing safety byte built into `payload` matches the reference layout.
fn emit(minbits: u32, minval: u64, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + payload.len().max(1));
    write_header(&mut out, minbits, minval);
    if payload.is_empty() {
        // minbits == 0: the reference still reserves one trailing byte.
        out.push(0);
    } else {
        out.extend_from_slice(payload);
    }
    out
}

/// Assemble a full-precision (raw) chunk: header + verbatim element bytes.
fn emit_raw(full_bits: u32, minval: u64, input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + input.len());
    write_header(&mut out, full_bits, minval);
    out.extend_from_slice(input);
    out
}

fn write_header(out: &mut Vec<u8>, minbits: u32, minval: u64) {
    out.extend_from_slice(&minbits.to_le_bytes()); // bytes 0..4
    out.push(8); // byte 4: sizeof(minval)
    out.extend_from_slice(&minval.to_le_bytes()); // bytes 5..13
    out.extend_from_slice(&[0u8; HEADER_LEN - 13]); // bytes 13..21: padding
}

// --- bit packing ----------------------------------------------------------

/// Pack `nelmts` offsets, `minbits` bits each, MSB-first. The buffer length
/// matches the reference `nelmts * minbits / 8 + 1`: enough for the bit
/// stream plus, when the stream is a multiple of 8 bits, one trailing zero
/// the reference always reserves.
///
/// Implementation maintains a `u64` accumulator with the most recently
/// written bits at its bottom, flushing whole bytes from the top as the
/// accumulator fills. This processes a chunk of `nelmts * minbits` bits
/// with O(nelmts) iterations rather than O(nelmts * minbits) — roughly an
/// order of magnitude fewer instructions in the hot path for typical
/// minbits.
fn pack_offsets(offsets: &[u64], minbits: u32, nelmts: usize) -> Result<Vec<u8>, FormatError> {
    // `nelmts * minbits` is a bit count that sizes the payload buffer. Compute
    // it in `u64` (cannot overflow there: `nelmts` originates from a `u32` and
    // `minbits` is <= 64) and narrow the resulting byte length to `usize`,
    // erroring instead of truncating on a 32-bit host.
    let payload_len = (nelmts as u64 * minbits as u64 / 8 + 1).to_usize()?;
    if minbits == 0 {
        // All offsets are zero; the reference layout is just `payload_len`
        // zero bytes.
        return Ok(vec![0u8; payload_len]);
    }
    let mut buf = Vec::with_capacity(payload_len);
    let mask: u64 = if minbits == 64 {
        u64::MAX
    } else {
        (1u64 << minbits) - 1
    };
    let mut acc: u64 = 0;
    let mut nbits: u32 = 0;
    for &v in offsets {
        // Drain whole bytes already buffered so nbits ∈ 0..=7 going in.
        while nbits >= 8 {
            nbits -= 8;
            buf.push(((acc >> nbits) & 0xFF) as u8);
        }
        // Merge in `minbits` bits. Work in u128 because `acc << 64` would be
        // UB in u64 (we'd hit it when minbits == 64 and nbits == 0).
        let combined = ((acc as u128) << minbits) | (v & mask) as u128;
        let total = nbits + minbits;
        if total <= 64 {
            acc = combined as u64;
            nbits = total;
        } else {
            // The merged value spans more than a u64; emit 8 bytes from the
            // top and keep the spillover (always 1..=7 bits) for next round.
            let drop = total - 64;
            let top = (combined >> drop) as u64;
            buf.extend_from_slice(&top.to_be_bytes());
            acc = (combined & ((1u128 << drop) - 1)) as u64;
            nbits = drop;
        }
    }
    while nbits >= 8 {
        nbits -= 8;
        buf.push(((acc >> nbits) & 0xFF) as u8);
    }
    if nbits > 0 {
        // Left-align the remaining bits into a final byte; the low bits are
        // zero padding within the partial byte.
        buf.push(((acc << (8 - nbits)) & 0xFF) as u8);
    }
    // For bit streams that end on a byte boundary, the reference still
    // reserves one trailing zero. The formula above sized the buffer for
    // either case; pad here.
    while buf.len() < payload_len {
        buf.push(0);
    }
    Ok(buf)
}

/// Streaming MSB-first bit reader over a packed payload. Fused with the
/// reconstruction loop so we don't have to materialize a `Vec<u64>` of
/// intermediate offsets.
struct BitReader<'a> {
    payload: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(payload: &'a [u8]) -> Self {
        Self {
            payload,
            bit_pos: 0,
        }
    }

    /// Read the next `minbits` bits (1..=64) MSB-first. Caller validates
    /// that the payload holds enough bits up front.
    #[inline]
    fn read(&mut self, minbits: u32) -> u64 {
        debug_assert!((1..=64).contains(&minbits));
        let byte = self.bit_pos >> 3;
        let off = (self.bit_pos & 7) as u32;

        // Load a 72-bit window at `byte`. Hot path: payload has ≥ 9 bytes
        // remaining and we can read directly. Tail: zero-pad a stack buffer.
        let (hi, lo) = if byte + 9 <= self.payload.len() {
            let hi = u64::from_be_bytes(self.payload[byte..byte + 8].try_into().unwrap());
            let lo = self.payload[byte + 8] as u64;
            (hi, lo)
        } else {
            let mut window = [0u8; 9];
            let take = self.payload.len().saturating_sub(byte).min(9);
            window[..take].copy_from_slice(&self.payload[byte..byte + take]);
            let hi = u64::from_be_bytes(window[..8].try_into().unwrap());
            let lo = window[8] as u64;
            (hi, lo)
        };

        // Align the window so the desired `minbits` bits sit at the top of
        // a 64-bit value, then shift them down. The off == 0 branch avoids
        // the otherwise-suspicious `lo >> 8` on a value with only 8 set bits.
        let combined = if off == 0 {
            hi
        } else {
            (hi << off) | (lo >> (8 - off))
        };
        self.bit_pos += minbits as usize;
        combined >> (64 - minbits)
    }
}

/// Test-only helper: collect a full bitstream into a `Vec<u64>`. The main
/// decompress path drives the reconstructor directly off [`BitReader`].
#[cfg(test)]
fn unpack_bits(payload: &[u8], nelmts: usize, minbits: u32) -> Result<Vec<u64>, FormatError> {
    let total_bits = nelmts * minbits as usize;
    if payload.len() * 8 < total_bits {
        return Err(FormatError::FilterError(
            "scaleoffset: payload too short for packed data".to_string(),
        ));
    }
    let mut reader = BitReader::new(payload);
    Ok((0..nelmts).map(|_| reader.read(minbits)).collect())
}

// --- value (de)serialization ----------------------------------------------

/// Read a `size`-byte element as a u64 (low `size` bytes meaningful),
/// normalizing to little-endian regardless of stored `order`.
fn read_value(chunk: &[u8], size: usize, order: u32) -> u64 {
    let mut bytes = [0u8; 8];
    if order == ORDER_LE {
        bytes[..size].copy_from_slice(&chunk[..size]);
    } else {
        for (k, &b) in chunk[..size].iter().enumerate() {
            bytes[size - 1 - k] = b;
        }
    }
    u64::from_le_bytes(bytes)
}

/// Write the low `size` bytes of `bits` in the dataset's byte order.
fn write_value(out: &mut Vec<u8>, bits: u64, size: usize, order: u32) {
    let le = bits.to_le_bytes();
    if order == ORDER_LE {
        out.extend_from_slice(&le[..size]);
    } else {
        for k in (0..size).rev() {
            out.push(le[k]);
        }
    }
}

fn sign_extend(bits: u64, size: usize) -> i64 {
    if size >= 8 {
        bits as i64
    } else {
        let shift = 64 - size * 8;
        ((bits << shift) as i64) >> shift
    }
}

/// `1 << minbits - 1`, the all-ones offset that flags a fill value.
fn sentinel(minbits: u32) -> u64 {
    (1u64 << minbits).wrapping_sub(1)
}

/// Reassemble a `size`-byte fill value from `cd_values[8..]` (stored
/// least-significant 4 bytes per entry).
fn read_fill_bits(cd: &[u32], size: usize) -> Result<u64, FormatError> {
    let entries = size.div_ceil(4);
    if cd.len() < PARM_FILVAL + entries {
        return Err(FormatError::FilterError(
            "scaleoffset: cd_values missing fill value".to_string(),
        ));
    }
    let mut bytes = [0u8; 8];
    let mut off = 0;
    let mut idx = PARM_FILVAL;
    while off < size {
        let take = (size - off).min(4);
        bytes[off..off + take].copy_from_slice(&cd[idx].to_le_bytes()[..take]);
        off += take;
        idx += 1;
    }
    Ok(u64::from_le_bytes(bytes))
}

/// `10^exp` as `f64`, computed without `std` (the float `powf`/`powi` methods
/// require `std`). `exp` is a small decimal scale factor; exponentiation by
/// squaring keeps this both cheap and accurate.
fn pow10_f64(exp: i32) -> f64 {
    let mut result = 1.0f64;
    let mut base = 10.0f64;
    let mut n = exp.unsigned_abs();
    while n > 0 {
        if n & 1 == 1 {
            result *= base;
        }
        base *= base;
        n >>= 1;
    }
    if exp < 0 { 1.0 / result } else { result }
}

/// `10^exp` as `f32` (computed in `f64` for accuracy, then narrowed).
fn pow10_f32(exp: i32) -> f32 {
    pow10_f64(exp) as f32
}

/// Round half away from zero to the nearest integer, matching C `llround`.
/// Float-to-int `as` casts saturate in Rust, so out-of-range inputs are safe.
fn round_half_away_f64(x: f64) -> i64 {
    if x >= 0.0 {
        (x + 0.5) as i64
    } else {
        (x - 0.5) as i64
    }
}

/// `f32` counterpart of [`round_half_away_f64`] (matches C `lroundf`).
fn round_half_away_f32(x: f32) -> i64 {
    if x >= 0.0 {
        (x + 0.5) as i64
    } else {
        (x - 0.5) as i64
    }
}

/// Ceiling of log2, matching the reference `H5Z__scaleoffset_log2`
/// (`log2(0) == 1`, `log2(1) == 0`).
fn ceil_log2(num: u64) -> u32 {
    let mut v = 0u32;
    let mut lower_bound = 1u64;
    let mut val = num;
    loop {
        val >>= 1;
        if val == 0 {
            break;
        }
        v += 1;
        lower_bound <<= 1;
    }
    if num == lower_bound { v } else { v + 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_filter(size: u32, signed: bool, order: u32, nelmts: u32) -> FilterDescription {
        let ty = ScaleOffsetType {
            class: CLS_INTEGER,
            sign: if signed { SGN_2 } else { SGN_NONE },
            order,
        };
        FilterDescription {
            filter_id: crate::filter_pipeline::FILTER_SCALEOFFSET,
            name: None,
            flags: 0,
            client_data: build_cd_values(ScaleOffset::Integer(0), ty, size, nelmts).unwrap(),
        }
    }

    fn float_filter(size: u32, decimals: i32, order: u32, nelmts: u32) -> FilterDescription {
        let ty = ScaleOffsetType {
            class: CLS_FLOAT,
            sign: SGN_NONE,
            order,
        };
        FilterDescription {
            filter_id: crate::filter_pipeline::FILTER_SCALEOFFSET,
            name: None,
            flags: 0,
            client_data: build_cd_values(ScaleOffset::FloatDScale(decimals), ty, size, nelmts)
                .unwrap(),
        }
    }

    #[test]
    fn ceil_log2_matches_reference() {
        assert_eq!(ceil_log2(0), 1);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(255), 8);
        assert_eq!(ceil_log2(256), 8);
        assert_eq!(ceil_log2(257), 9);
    }

    fn roundtrip_u32(vals: &[u32], order: u32) {
        let mut raw = Vec::new();
        for &v in vals {
            if order == ORDER_LE {
                raw.extend_from_slice(&v.to_le_bytes());
            } else {
                raw.extend_from_slice(&v.to_be_bytes());
            }
        }
        let f = int_filter(4, false, order, vals.len() as u32);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f).unwrap();
        assert_eq!(dec, raw);
    }

    #[test]
    fn integer_unsigned_roundtrip_le_and_be() {
        let vals = [100u32, 105, 101, 110, 100, 128];
        roundtrip_u32(&vals, ORDER_LE);
        roundtrip_u32(&vals, ORDER_BE);
    }

    #[test]
    fn integer_signed_roundtrip_with_negatives() {
        let vals: [i16; 6] = [-100, -50, -100, 0, 27, -99];
        for &order in &[ORDER_LE, ORDER_BE] {
            let mut raw = Vec::new();
            for &v in &vals {
                if order == ORDER_LE {
                    raw.extend_from_slice(&v.to_le_bytes());
                } else {
                    raw.extend_from_slice(&v.to_be_bytes());
                }
            }
            let f = int_filter(2, true, order, vals.len() as u32);
            let comp = compress(&raw, &f).unwrap();
            let dec = decompress(&comp, &f).unwrap();
            assert_eq!(dec, raw, "order {order}");
        }
    }

    #[test]
    fn integer_all_equal_uses_minbits_zero() {
        let vals = [7u32; 5];
        let mut raw = Vec::new();
        for &v in &vals {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let f = int_filter(4, false, ORDER_LE, vals.len() as u32);
        let comp = compress(&raw, &f).unwrap();
        // minbits == 0 -> 21-byte header + 1 trailing byte.
        assert_eq!(comp.len(), HEADER_LEN + 1);
        assert_eq!(u32::from_le_bytes([comp[0], comp[1], comp[2], comp[3]]), 0);
        let dec = decompress(&comp, &f).unwrap();
        assert_eq!(dec, raw);
    }

    #[test]
    fn integer_full_range_uses_raw_path() {
        // 0 and u32::MAX force the full-precision (raw) path.
        let vals = [0u32, u32::MAX, 123];
        let mut raw = Vec::new();
        for &v in &vals {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let f = int_filter(4, false, ORDER_LE, vals.len() as u32);
        let comp = compress(&raw, &f).unwrap();
        assert_eq!(comp.len(), HEADER_LEN + raw.len());
        assert_eq!(u32::from_le_bytes([comp[0], comp[1], comp[2], comp[3]]), 32);
        let dec = decompress(&comp, &f).unwrap();
        assert_eq!(dec, raw);
    }

    #[test]
    fn integer_u8_roundtrip() {
        let raw = vec![10u8, 11, 12, 250, 10, 200];
        let f = int_filter(1, false, ORDER_LE, raw.len() as u32);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f).unwrap();
        assert_eq!(dec, raw);
    }

    #[test]
    fn integer_i64_roundtrip() {
        let vals: [i64; 4] = [-1_000_000, 5, -999_999, 42];
        let mut raw = Vec::new();
        for &v in &vals {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let f = int_filter(8, true, ORDER_LE, vals.len() as u32);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f).unwrap();
        assert_eq!(dec, raw);
    }

    #[test]
    fn float_dscale_roundtrip_within_tolerance() {
        let vals = [1.234f64, 1.235, 1.250, 1.111, 1.234, 1.999];
        let decimals = 3;
        let mut raw = Vec::new();
        for &v in &vals {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let f = float_filter(8, decimals, ORDER_LE, vals.len() as u32);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f).unwrap();
        let got: Vec<f64> = dec
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let tol = 0.5 * 10f64.powi(-decimals);
        for (g, v) in got.iter().zip(vals.iter()) {
            assert!((g - v).abs() <= tol, "got {g}, want {v}");
        }
    }

    #[test]
    fn float32_dscale_roundtrip_be() {
        let vals = [10.25f32, 10.50, 10.75, 10.00, 10.25];
        let decimals = 2;
        let mut raw = Vec::new();
        for &v in &vals {
            raw.extend_from_slice(&v.to_be_bytes());
        }
        let f = float_filter(4, decimals, ORDER_BE, vals.len() as u32);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f).unwrap();
        let got: Vec<f32> = dec
            .chunks_exact(4)
            .map(|c| f32::from_be_bytes(c.try_into().unwrap()))
            .collect();
        let tol = 0.5 * 10f32.powi(-decimals);
        for (g, v) in got.iter().zip(vals.iter()) {
            assert!((g - v).abs() <= tol, "got {g}, want {v}");
        }
    }

    #[test]
    fn truncated_chunk_errors_not_panics() {
        // A chunk whose header claims minbits=3 but is shorter than the 21-byte
        // header region must error rather than panic when slicing the payload.
        let f = int_filter(4, false, ORDER_LE, 4);
        let mut bad = Vec::new();
        bad.extend_from_slice(&3u32.to_le_bytes()); // minbits = 3
        bad.push(8); // minval size
        bad.extend_from_slice(&0u64.to_le_bytes()); // minval (bytes 5..13)
        // Only 13 bytes total: shorter than the 21-byte header.
        assert!(matches!(
            decompress(&bad, &f),
            Err(FormatError::FilterError(_))
        ));
    }

    #[test]
    fn header_byte_layout() {
        // Two u32 values {min=5, max=9} -> span 5 -> minbits 3.
        let vals = [5u32, 9, 6, 5];
        let mut raw = Vec::new();
        for &v in &vals {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let f = int_filter(4, false, ORDER_LE, vals.len() as u32);
        let comp = compress(&raw, &f).unwrap();
        assert_eq!(u32::from_le_bytes([comp[0], comp[1], comp[2], comp[3]]), 3);
        assert_eq!(comp[4], 8); // sizeof(minval)
        let minval = u64::from_le_bytes(comp[5..13].try_into().unwrap());
        assert_eq!(minval, 5);
        assert_eq!(&comp[13..21], &[0u8; 8]); // padding
    }

    #[test]
    fn build_cd_values_rejects_mismatched_mode() {
        let int_ty = ScaleOffsetType {
            class: CLS_INTEGER,
            sign: SGN_2,
            order: ORDER_LE,
        };
        assert!(build_cd_values(ScaleOffset::FloatDScale(2), int_ty, 4, 10).is_err());
        let float_ty = ScaleOffsetType {
            class: CLS_FLOAT,
            sign: SGN_NONE,
            order: ORDER_LE,
        };
        assert!(build_cd_values(ScaleOffset::Integer(0), float_ty, 8, 10).is_err());
    }

    /// Deterministic xorshift64 used to drive the property-style tests below.
    /// Keeps the dev-dependency footprint minimal while still covering a wide
    /// input space; seeds are fixed so failures are reproducible.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn range(&mut self, hi: u64) -> u64 {
            // Modulo bias is fine for our coverage purposes.
            self.next() % hi
        }
    }

    #[test]
    fn pack_unpack_equivalence_random() {
        // For every minbits in 1..=64 and across many random widths/lengths,
        // pack(offsets) followed by unpack must return the original offsets.
        let mut rng = Rng::new(0x00C0_FFEE_F00D_1234);
        for _ in 0..400 {
            let minbits = (rng.range(64) + 1) as u32; // 1..=64
            let nelmts = (rng.range(257) + 1) as usize; // 1..=257
            let mask = if minbits == 64 {
                u64::MAX
            } else {
                (1u64 << minbits) - 1
            };
            let offsets: Vec<u64> = (0..nelmts).map(|_| rng.next() & mask).collect();
            let packed = pack_offsets(&offsets, minbits, nelmts).unwrap();
            // Reference layout reserves at least one trailing byte.
            assert_eq!(packed.len(), nelmts * minbits as usize / 8 + 1);
            let unpacked = unpack_bits(&packed, nelmts, minbits).unwrap();
            assert_eq!(
                unpacked, offsets,
                "minbits={minbits}, nelmts={nelmts}, seed=0xC0FFEEF00D1234"
            );
        }
    }

    fn roundtrip_random<T: Copy>(
        seed: u64,
        size: u32,
        signed: bool,
        order: u32,
        encode: impl Fn(&mut Rng) -> T,
        to_bytes: impl Fn(T, u32) -> Vec<u8>,
    ) {
        let mut rng = Rng::new(seed);
        for trial in 0..40 {
            let nelmts = (rng.range(199) + 1) as usize;
            let mut raw = Vec::with_capacity(nelmts * size as usize);
            for _ in 0..nelmts {
                raw.extend_from_slice(&to_bytes(encode(&mut rng), order));
            }
            let f = int_filter(size, signed, order, nelmts as u32);
            let comp = compress(&raw, &f).unwrap();
            let dec = decompress(&comp, &f).unwrap();
            assert_eq!(
                dec, raw,
                "trial {trial}: size={size}, signed={signed}, order={order}"
            );
        }
    }

    #[test]
    fn integer_roundtrip_random_u8() {
        roundtrip_random::<u8>(0x11, 1, false, ORDER_LE, |r| r.next() as u8, |v, _| vec![v]);
    }

    #[test]
    fn integer_roundtrip_random_u16_le_and_be() {
        for &order in &[ORDER_LE, ORDER_BE] {
            roundtrip_random::<u16>(
                0x22 ^ order as u64,
                2,
                false,
                order,
                |r| r.next() as u16,
                |v, o| {
                    if o == ORDER_LE {
                        v.to_le_bytes().to_vec()
                    } else {
                        v.to_be_bytes().to_vec()
                    }
                },
            );
        }
    }

    #[test]
    fn integer_roundtrip_random_i32_narrow_and_wide() {
        // Mix narrow (small span -> small minbits) and wide (forces raw path).
        let mut rng = Rng::new(0x33);
        for trial in 0..40 {
            let nelmts = (rng.range(199) + 1) as usize;
            let narrow = rng.next() & 1 == 0;
            let mut raw = Vec::with_capacity(nelmts * 4);
            for _ in 0..nelmts {
                let v: i32 = if narrow {
                    (rng.range(1024) as i32) - 512 // span ~= 1024
                } else {
                    rng.next() as i32 // full range
                };
                raw.extend_from_slice(&v.to_le_bytes());
            }
            let f = int_filter(4, true, ORDER_LE, nelmts as u32);
            let comp = compress(&raw, &f).unwrap();
            let dec = decompress(&comp, &f).unwrap();
            assert_eq!(dec, raw, "trial {trial}: narrow={narrow}");
        }
    }

    #[test]
    fn integer_roundtrip_random_i64() {
        roundtrip_random::<i64>(
            0x44,
            8,
            true,
            ORDER_LE,
            |r| {
                // Bias toward narrow spans so we exercise the packing path,
                // but include the full-range tail too.
                let span = 1u64 << (r.range(48) as u32 + 1);
                let v = (r.next() % span) as i64;
                let off = (r.next() as i64) >> 1;
                v.wrapping_add(off)
            },
            |v, _| v.to_le_bytes().to_vec(),
        );
    }

    #[test]
    fn float_dscale_roundtrip_random_within_tolerance() {
        let mut rng = Rng::new(0x55);
        for trial in 0..30 {
            let nelmts = (rng.range(99) + 1) as usize;
            let decimals = (rng.range(5) as i32) + 1; // 1..=5
            // Keep magnitudes modest so f64 ULP stays well below the D-scale
            // tolerance. Avoid generating values pre-quantized to the same
            // precision (decimals) we then round to, otherwise the input
            // sits exactly on a rounding boundary.
            let mut raw = Vec::with_capacity(nelmts * 8);
            let mut vals = Vec::with_capacity(nelmts);
            for _ in 0..nelmts {
                let v = (rng.next() as i32 as f64) * 1e-9; // |v| <= ~2.1
                vals.push(v);
                raw.extend_from_slice(&v.to_le_bytes());
            }
            let f = float_filter(8, decimals, ORDER_LE, nelmts as u32);
            let comp = compress(&raw, &f).unwrap();
            let dec = decompress(&comp, &f).unwrap();
            let got: Vec<f64> = dec
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            // 0.5 ULP rounding + a few ULP of float arithmetic noise.
            let tol = 0.501 * 10f64.powi(-decimals);
            for (g, v) in got.iter().zip(vals.iter()) {
                assert!(
                    (g - v).abs() <= tol,
                    "trial {trial}: got {g}, want {v} (tol {tol})"
                );
            }
        }
    }

    /// Build a synthetic chunk where `minbits == 0`, the compressed header
    /// declares `minval = 7`, but the filter's `cd_values` say
    /// `filavail = FILL_DEFINED` with `filval = 99`. The reference C decoder
    /// short-circuits the `minbits == 0` path to emit `minval` for every
    /// element. The Rust port must match: every element should reconstruct
    /// to 7, not 99. Catches the sentinel-collision divergence flagged in
    /// the scale-offset review.
    #[test]
    fn minbits_zero_with_fill_defined_emits_minval_not_filval() {
        let nelmts = 5u32;
        let size = 4u32;
        let mut cd = vec![0u32; TOTAL_NPARMS];
        cd[PARM_SCALETYPE] = SO_INT;
        cd[PARM_SCALEFACTOR] = 0;
        cd[PARM_NELMTS] = nelmts;
        cd[PARM_CLASS] = CLS_INTEGER;
        cd[PARM_SIZE] = size;
        cd[PARM_SIGN] = SGN_NONE;
        cd[PARM_ORDER] = ORDER_LE;
        cd[PARM_FILAVAIL] = FILL_DEFINED;
        cd[PARM_FILVAL] = 99;
        let f = FilterDescription {
            filter_id: crate::filter_pipeline::FILTER_SCALEOFFSET,
            name: None,
            flags: 0,
            client_data: cd,
        };
        // 21-byte header (minbits=0, minval=7) + 1-byte trailing pad.
        let mut chunk = Vec::with_capacity(HEADER_LEN + 1);
        chunk.extend_from_slice(&0u32.to_le_bytes());
        chunk.push(8);
        chunk.extend_from_slice(&7u64.to_le_bytes());
        chunk.extend_from_slice(&[0u8; HEADER_LEN - 13]);
        chunk.push(0);
        let out = decompress(&chunk, &f).unwrap();
        let got: Vec<u32> = out
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(
            got,
            vec![7u32; nelmts as usize],
            "minbits==0 with FILL_DEFINED must emit minval, not filval"
        );
    }

    /// A chunk with `minbits > 0` and `FILL_DEFINED`: the offsets at the
    /// all-ones sentinel must reconstruct to `filval`, while every other
    /// offset reconstructs to `minval + offset`. Companion to the
    /// `minbits == 0` regression above — together they cover both sides of
    /// the sentinel branch.
    #[test]
    fn fill_defined_sentinel_emits_filval_not_minval() {
        let nelmts = 4u32;
        let size = 4u32;
        let mut cd = vec![0u32; TOTAL_NPARMS];
        cd[PARM_SCALETYPE] = SO_INT;
        cd[PARM_SCALEFACTOR] = 0;
        cd[PARM_NELMTS] = nelmts;
        cd[PARM_CLASS] = CLS_INTEGER;
        cd[PARM_SIZE] = size;
        cd[PARM_SIGN] = SGN_NONE;
        cd[PARM_ORDER] = ORDER_LE;
        cd[PARM_FILAVAIL] = FILL_DEFINED;
        cd[PARM_FILVAL] = 999;
        let f = FilterDescription {
            filter_id: crate::filter_pipeline::FILTER_SCALEOFFSET,
            name: None,
            flags: 0,
            client_data: cd,
        };
        // minbits = 3 → sentinel = 7. Offsets [0, 1, 7, 2] expect
        // [minval+0, minval+1, filval, minval+2] = [10, 11, 999, 12].
        let mut chunk = Vec::new();
        chunk.extend_from_slice(&3u32.to_le_bytes());
        chunk.push(8);
        chunk.extend_from_slice(&10u64.to_le_bytes());
        chunk.extend_from_slice(&[0u8; HEADER_LEN - 13]);
        chunk.extend_from_slice(&pack_offsets(&[0, 1, 7, 2], 3, 4).unwrap());
        let out = decompress(&chunk, &f).unwrap();
        let got: Vec<u32> = out
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, vec![10u32, 11, 999, 12]);
    }

    /// The writer doesn't support encoding with a defined fill value (the
    /// builder doesn't expose one). If someone hands it a `FILL_DEFINED`
    /// `cd_values` directly, it must error rather than silently miscompress.
    #[test]
    fn compress_rejects_fill_defined() {
        let nelmts = 4u32;
        let size = 4u32;
        let mut cd = vec![0u32; TOTAL_NPARMS];
        cd[PARM_SCALETYPE] = SO_INT;
        cd[PARM_NELMTS] = nelmts;
        cd[PARM_CLASS] = CLS_INTEGER;
        cd[PARM_SIZE] = size;
        cd[PARM_SIGN] = SGN_NONE;
        cd[PARM_ORDER] = ORDER_LE;
        cd[PARM_FILAVAIL] = FILL_DEFINED;
        cd[PARM_FILVAL] = 0;
        let f = FilterDescription {
            filter_id: crate::filter_pipeline::FILTER_SCALEOFFSET,
            name: None,
            flags: 0,
            client_data: cd,
        };
        let raw = vec![0u8; nelmts as usize * size as usize];
        assert!(matches!(
            compress(&raw, &f),
            Err(FormatError::FilterError(_))
        ));
    }

    /// Float E-scale is forbidden by the reference library too; we error
    /// before touching the payload.
    #[test]
    fn decompress_rejects_escale() {
        let mut cd = vec![0u32; TOTAL_NPARMS];
        cd[PARM_SCALETYPE] = SO_FLOAT_ESCALE;
        cd[PARM_NELMTS] = 4;
        cd[PARM_CLASS] = CLS_FLOAT;
        cd[PARM_SIZE] = 4;
        cd[PARM_ORDER] = ORDER_LE;
        let f = FilterDescription {
            filter_id: crate::filter_pipeline::FILTER_SCALEOFFSET,
            name: None,
            flags: 0,
            client_data: cd,
        };
        let chunk = vec![0u8; HEADER_LEN + 4];
        assert!(matches!(
            decompress(&chunk, &f),
            Err(FormatError::FilterError(_))
        ));
    }

    /// A corrupt header that claims `minbits > size * 8` must error rather
    /// than reach the bit reader and panic.
    #[test]
    fn decompress_rejects_oversized_minbits_header() {
        let f = int_filter(4, false, ORDER_LE, 4);
        // size=4 → full_bits=32. Set minbits=33.
        let mut bad = Vec::new();
        bad.extend_from_slice(&33u32.to_le_bytes());
        bad.push(8);
        bad.extend_from_slice(&0u64.to_le_bytes());
        bad.extend_from_slice(&[0u8; HEADER_LEN - 13]);
        bad.extend_from_slice(&[0u8; 16]); // dummy payload
        assert!(matches!(
            decompress(&bad, &f),
            Err(FormatError::FilterError(_))
        ));
    }

    /// Signed 1-byte ints: covers the `size == 1` arm of `sign_extend`,
    /// which the unsigned `u8` round-trip skips.
    #[test]
    fn integer_i8_roundtrip_with_negatives() {
        let vals: [i8; 6] = [-100, -50, 0, 27, -99, 100];
        let raw: Vec<u8> = vals.iter().map(|&v| v as u8).collect();
        let f = int_filter(1, true, ORDER_LE, vals.len() as u32);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f).unwrap();
        assert_eq!(dec, raw);
    }

    /// `nelmts == 1` exercises the streaming min/max loops in
    /// `precompress_{integer,float}` with an empty body (the `1..nelmts`
    /// range yields nothing). Span is 0, minbits is 0 — round-trips both
    /// integer and float through the all-equal short-circuit.
    #[test]
    fn single_element_chunk_roundtrip() {
        let raw = 42u32.to_le_bytes().to_vec();
        let f = int_filter(4, false, ORDER_LE, 1);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f).unwrap();
        assert_eq!(dec, raw);

        let raw = 3.14f64.to_le_bytes().to_vec();
        let f = float_filter(8, 3, ORDER_LE, 1);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f).unwrap();
        let got = f64::from_le_bytes(dec.as_slice().try_into().unwrap());
        assert!((got - 3.14).abs() <= 0.5e-3);
    }

    #[test]
    fn scale_offset_type_from_datatype_classes() {
        let i32_ty = Datatype::FixedPoint {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        };
        let so = scale_offset_type_from_datatype(&i32_ty).unwrap();
        assert_eq!(so.class, CLS_INTEGER);
        assert_eq!(so.sign, SGN_2);
        assert_eq!(so.order, ORDER_LE);

        let f64_ty = Datatype::FloatingPoint {
            size: 8,
            byte_order: DatatypeByteOrder::BigEndian,
            bit_offset: 0,
            bit_precision: 64,
            exponent_location: 52,
            exponent_size: 11,
            mantissa_location: 0,
            mantissa_size: 52,
            exponent_bias: 1023,
        };
        let so = scale_offset_type_from_datatype(&f64_ty).unwrap();
        assert_eq!(so.class, CLS_FLOAT);
        assert_eq!(so.order, ORDER_BE);
    }
}
