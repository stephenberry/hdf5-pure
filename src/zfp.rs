//! Pure-Rust ZFP fixed-rate codec for f32, f64, i32, i64.
//!
//! Direct port of the reference LLNL/zfp C algorithm
//! (<https://github.com/LLNL/zfp>), currently specialized to 1D blocks of 4
//! values. Interoperable byte-for-byte with the reference bitstream: this is
//! enforced by `tests/zfp_crosscheck.rs`, which compares output against
//! fixtures produced by the real H5Z-ZFP plugin and zfpy.
//!
//! # Algorithm per block (4 values for 1D)
//!
//! **Float (f32/f64)**:
//!   1. Block-floating-point cast: compute `emax = max frexp-exponent` across
//!      the block; scale each value by `2^(PBITS - 2 - emax)` and truncate to
//!      a signed integer of width `PBITS` (32 or 64 bits).
//!   2. Emit a block header: `1` bit for non-empty, then `EBITS` bits for
//!      `emax + EBIAS` (8 bits / EBIAS=127 for f32; 11 / 1023 for f64). An
//!      empty all-zero block emits a single `0` bit and pads to `maxbits`.
//!
//! **Integer (i32/i64)**: skip the float header and cast; use the values
//! directly as signed integers.
//!
//! **Then, common to both**:
//!   3. Forward lifting transform: 5-stage non-orthogonal decorrelating
//!      transform on the 4 signed integers.
//!   4. Negabinary conversion: `(x + NBMASK) ^ NBMASK` per coefficient, where
//!      `NBMASK = 0xAA..AA` of the coefficient width. Maps two's-complement
//!      signed to unsigned such that the bit-plane encoder treats sign
//!      uniformly.
//!   5. Embedded bit-plane encoding: MSB-to-LSB, for each plane emit
//!      refinement bits for coefficients already marked significant, then run
//!      a unary run-length scan over the remaining coefficients (group-test
//!      bit; if positive, scan one-by-one until a 1-bit is found; if only one
//!      remains after a positive group test, its significance is implicit).
//!   6. Pad to exactly `rate * block_size` bits.
//!
//! The bit stream uses 64-bit words in **little-endian** byte order with bits
//! packed LSB-first within each word.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::error::FormatError;

/// Scalar type the codec operates on.
///
/// The H5Z-ZFP plugin encodes this as a small integer inside cd_values;
/// Step 6 wires the cd_values decoder up to populate this from a stored
/// filter on read. For write, the dataset's datatype determines which
/// variant is passed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZfpElementType {
    F32,
    F64,
    I32,
    I64,
}

/// Number of values per ZFP block (1D).
const BLOCK_SIZE: usize = 4;

/// Exponent bits in the block-float header, per scalar type.
const EBITS_F32: u32 = 8;
const EBITS_F64: u32 = 11;

/// Negabinary masks (per integer width).
const NBMASK_U32: u32 = 0xAAAA_AAAA;
const NBMASK_U64: u64 = 0xAAAA_AAAA_AAAA_AAAA;

// ---------------------------------------------------------------------------
// Bit-stream I/O
// ---------------------------------------------------------------------------

/// A bit-level writer that packs bits LSB-first into 64-bit little-endian words.
struct BitWriter {
    /// Completed 64-bit words (stored as raw little-endian bytes).
    buf: Vec<u8>,
    /// Current partial word being filled.
    word: u64,
    /// Number of valid bits in `word` (0..64).
    bits: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            word: 0,
            bits: 0,
        }
    }

    /// Write the `n` least-significant bits of `value`. Bits above `n` in
    /// `value` are ignored (masked off) so the caller may pass a packed
    /// bit-plane register without pre-masking.
    #[inline]
    fn write(&mut self, n: u32, value: u64) {
        debug_assert!(n <= 64);
        if n == 0 {
            return;
        }
        let masked = if n == 64 {
            value
        } else {
            value & ((1u64 << n) - 1)
        };
        self.word |= masked << self.bits;
        self.bits += n;
        if self.bits >= 64 {
            self.buf.extend_from_slice(&self.word.to_le_bytes());
            self.bits -= 64;
            // Carry: the high bits of `masked` that didn't fit in the
            // flushed word. When bits was 0 and n == 64 we've already
            // flushed everything, so a fresh word starts at 0.
            self.word = if self.bits > 0 {
                masked >> (n - self.bits)
            } else {
                0
            };
        }
    }

    /// Write a single bit.
    #[inline]
    fn write_bit(&mut self, bit: bool) {
        self.write(1, u64::from(bit));
    }

    /// Flush any partial word (zero-padded to 64 bits) and return the buffer.
    fn finish(mut self) -> Vec<u8> {
        if self.bits > 0 {
            self.buf.extend_from_slice(&self.word.to_le_bytes());
        }
        self.buf
    }

    /// Current position in bits.
    fn position(&self) -> usize {
        self.buf.len() * 8 + self.bits as usize
    }
}

/// A bit-level reader that unpacks bits LSB-first from 64-bit LE words.
struct BitReader<'a> {
    data: &'a [u8],
    /// Byte offset of the next word to load.
    byte_pos: usize,
    /// Current word.
    word: u64,
    /// Number of unconsumed bits remaining in `word`.
    bits: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut r = Self {
            data,
            byte_pos: 0,
            word: 0,
            bits: 0,
        };
        r.refill();
        r
    }

    /// Load the next 64-bit word from `data`.
    fn refill(&mut self) {
        if self.byte_pos + 8 <= self.data.len() {
            let bytes: [u8; 8] = self.data[self.byte_pos..self.byte_pos + 8]
                .try_into()
                .unwrap();
            self.word = u64::from_le_bytes(bytes);
            self.byte_pos += 8;
            self.bits = 64;
        } else {
            // Partial or exhausted -- zero-fill
            let mut bytes = [0u8; 8];
            let avail = self.data.len().saturating_sub(self.byte_pos);
            bytes[..avail].copy_from_slice(&self.data[self.byte_pos..self.byte_pos + avail]);
            self.word = u64::from_le_bytes(bytes);
            self.byte_pos += 8;
            self.bits = 64;
        }
    }

    /// Read `n` bits and return them right-justified.
    #[inline]
    fn read(&mut self, n: u32) -> u64 {
        debug_assert!(n <= 64);
        if n == 0 {
            return 0;
        }
        if n <= self.bits {
            let mask = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
            let val = self.word & mask;
            self.word >>= n;
            self.bits -= n;
            if self.bits == 0 {
                self.refill();
            }
            val
        } else {
            // Need bits from next word
            let lo_bits = self.bits;
            let lo = self.word; // all remaining bits
            self.refill();
            let hi_bits = n - lo_bits;
            let hi_mask = if hi_bits == 64 {
                u64::MAX
            } else {
                (1u64 << hi_bits) - 1
            };
            let hi = self.word & hi_mask;
            self.word >>= hi_bits;
            self.bits -= hi_bits;
            if self.bits == 0 {
                self.refill();
            }
            lo | (hi << lo_bits)
        }
    }

    /// Read a single bit.
    #[inline]
    fn read_bit(&mut self) -> bool {
        self.read(1) != 0
    }
}

// ---------------------------------------------------------------------------
// Block-floating-point cast (f32 <-> i32)
// ---------------------------------------------------------------------------

/// IEEE 754 f32 exponent bias.
const EBIAS_F32: i32 = 127;

/// Reference `pad_block` for a partial 1D block of width `n` (0 ≤ n ≤ 4).
///
/// For `n = 0` the block becomes all-zero. Otherwise the tail is filled by
/// the fall-through cascade: `p[1] = p[0]; p[2] = p[1]; p[3] = p[0]`
/// starting from `case n`. This matches the symmetric reflection used by
/// LLNL/zfp so that partial blocks encode identically under crosscheck.
macro_rules! impl_pad_partial {
    ($name:ident, $scalar:ty, $zero:expr) => {
        fn $name(block: &mut [$scalar; BLOCK_SIZE], n: usize) {
            match n {
                0 => {
                    block[0] = $zero;
                    block[1] = $zero;
                    block[2] = $zero;
                    block[3] = $zero;
                }
                1 => {
                    block[1] = block[0];
                    block[2] = block[1];
                    block[3] = block[0];
                }
                2 => {
                    block[2] = block[1];
                    block[3] = block[0];
                }
                3 => {
                    block[3] = block[0];
                }
                _ => {}
            }
        }
    };
}

impl_pad_partial!(pad_partial_block_f32, f32, 0.0f32);
impl_pad_partial!(pad_partial_block_f64, f64, 0.0f64);
impl_pad_partial!(pad_partial_block_i32, i32, 0i32);
impl_pad_partial!(pad_partial_block_i64, i64, 0i64);

/// Convert a block of 4 f32 values to block-floating-point i32 representation,
/// matching LLNL/zfp's `fwd_cast`.
///
/// The reference defines:
///
///   emax = max frexp-exponent across the block (0.5 ≤ |m| < 1 convention)
///   iblock[i] = (Int)(fblock[i] * 2^(30 - emax))
///
/// Using the IEEE biased exponent `b` of the largest-magnitude value,
/// frexp's exponent is `b - (EBIAS - 1) = b - 126`, so the scale becomes
/// `2^(156 - b)`. The returned header value is `emax_biased = frexp_emax +
/// EBIAS = b + 1`, which is what the reference stream writes (as the high
/// bits of `2*e + 1`).
///
/// If all values are zero or subnormal the reference's `exponent()` returns
/// `-EBIAS`, which we signal here as `emax_biased = 0` — the encoder then
/// emits a single zero-bit empty-block marker.
fn fwd_cast_f32(vals: &[f32; BLOCK_SIZE]) -> (u32, [i32; BLOCK_SIZE]) {
    let mut ieee_max: i32 = 0;
    for &v in vals {
        let e = ((v.to_bits() >> 23) & 0xFF) as i32;
        if e > ieee_max {
            ieee_max = e;
        }
    }
    if ieee_max == 0 {
        return (0, [0; BLOCK_SIZE]);
    }
    let emax_biased = (ieee_max + 1) as u32;
    // scale = 2^(30 - frexp_emax) = 2^(30 - (ieee_max - (EBIAS - 1)))
    //       = 2^(30 + EBIAS - 1 - ieee_max) = 2^(156 - ieee_max)
    let scale_exp = 30 + EBIAS_F32 - 1 - ieee_max;
    let scale = pow2_f64(scale_exp);
    let mut result = [0i32; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        // Do the multiply in f64 so subnormal / extreme scaling doesn't clip;
        // then truncate toward zero to match C's `(Int)` cast.
        let y = vals[i] as f64 * scale;
        result[i] = y as i32;
    }
    (emax_biased, result)
}

/// Inverse block-floating-point cast matching LLNL/zfp's `inv_cast`:
///
///   value = coefficient * 2^(emax - 30)   (frexp-exponent convention)
///
/// Given the stored header value `emax_biased = frexp_emax + EBIAS`, the
/// dequantization scale is `2^(emax_biased - EBIAS - 30)`.
fn inv_cast_f32(emax_biased: u32, coeffs: &[i32; BLOCK_SIZE]) -> [f32; BLOCK_SIZE] {
    let mut result = [0.0f32; BLOCK_SIZE];
    if emax_biased == 0 {
        return result;
    }
    let exp = emax_biased as i32 - EBIAS_F32 - 30;
    let scale = pow2_f64(exp);
    for (i, &c) in coeffs.iter().enumerate() {
        if c == 0 {
            continue;
        }
        result[i] = ((c as f64) * scale) as f32;
    }
    result
}

/// IEEE 754 f64 exponent bias.
const EBIAS_F64: i32 = 1023;

/// f64 analog of `fwd_cast_f32` — produces i64 coefficients scaled such that
/// `|coef| ≤ 2^62`, and the header value `emax_biased = frexp_emax + EBIAS`
/// (i.e., the raw IEEE biased exponent of the largest-magnitude input +1).
fn fwd_cast_f64(vals: &[f64; BLOCK_SIZE]) -> (u64, [i64; BLOCK_SIZE]) {
    let mut ieee_max: i32 = 0;
    for &v in vals {
        let e = ((v.to_bits() >> 52) & 0x7FF) as i32;
        if e > ieee_max {
            ieee_max = e;
        }
    }
    if ieee_max == 0 {
        return (0, [0; BLOCK_SIZE]);
    }
    let emax_biased = (ieee_max + 1) as u64;
    // scale = 2^(62 - frexp_emax). For f64: frexp_emax = ieee_max - 1022.
    // => scale_exp = 62 - (ieee_max - 1022) = 1084 - ieee_max.
    // ieee_max ∈ [1, 2046] ⇒ scale_exp ∈ [-962, 1083]; the upper end exceeds
    // the normal f64 range so use a two-step build.
    let scale = pow2_f64_wide(1084 - ieee_max);
    let mut result = [0i64; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        let y = vals[i] * scale;
        result[i] = y as i64; // saturating toward zero, same as C `(Int64)`
    }
    (emax_biased, result)
}

/// f64 inverse cast — `value = coef * 2^(emax_biased - EBIAS - 62)`.
fn inv_cast_f64(emax_biased: u64, coeffs: &[i64; BLOCK_SIZE]) -> [f64; BLOCK_SIZE] {
    let mut result = [0.0f64; BLOCK_SIZE];
    if emax_biased == 0 {
        return result;
    }
    let exp = emax_biased as i32 - EBIAS_F64 - 62;
    let scale = pow2_f64_wide(exp);
    for (i, &c) in coeffs.iter().enumerate() {
        if c == 0 {
            continue;
        }
        result[i] = (c as f64) * scale;
    }
    result
}

/// Like `pow2_f64` but handles `exp` outside `[-1022, 1023]` by splitting
/// into multiple steps. Needed for f64 quantization where the scale can
/// exceed the normal-range upper bound (up to ~2^1083).
#[inline]
fn pow2_f64_wide(exp: i32) -> f64 {
    if exp >= -1022 && exp <= 1023 {
        return pow2_f64(exp);
    }
    let mut remaining = exp;
    let mut acc = 1.0f64;
    while remaining > 1023 {
        acc *= pow2_f64(1023);
        remaining -= 1023;
    }
    while remaining < -1022 {
        acc *= pow2_f64(-1022);
        remaining += 1022;
    }
    acc * pow2_f64(remaining)
}

/// `2.0_f64.powi(exp)` without `std` / `libm`.
///
/// Builds the f64 directly from its IEEE 754 biased-exponent field. The ZFP
/// f32 codec uses `exp = emax - 127 - 30` where `emax ∈ [0, 255]`, giving
/// `exp ∈ [-157, 98]` — comfortably inside the normal f64 range — so we do
/// not need a subnormal/overflow fallback path.
#[inline]
fn pow2_f64(exp: i32) -> f64 {
    debug_assert!(exp >= -1022 && exp <= 1023, "pow2_f64 out of normal range");
    let biased = (exp + 1023) as u64;
    f64::from_bits(biased << 52)
}

// ---------------------------------------------------------------------------
// Lifting transform (1D, 4 values). Matches reference `fwd_lift` / `inv_lift`
// (`src/template/encode.c`, `decode.c`) — identical instruction sequence
// except for the integer width. Shifts are arithmetic (sign-preserving), so
// i32/i64 share the source.
// ---------------------------------------------------------------------------

macro_rules! impl_lift {
    ($fwd:ident, $inv:ident, $int:ty) => {
        #[allow(clippy::many_single_char_names)]
        fn $fwd(p: &mut [$int; BLOCK_SIZE]) {
            let mut x = p[0];
            let mut y = p[1];
            let mut z = p[2];
            let mut w = p[3];
            x = x.wrapping_add(w); x >>= 1; w = w.wrapping_sub(x);
            z = z.wrapping_add(y); z >>= 1; y = y.wrapping_sub(z);
            x = x.wrapping_add(z); x >>= 1; z = z.wrapping_sub(x);
            w = w.wrapping_add(y); w >>= 1; y = y.wrapping_sub(w);
            w = w.wrapping_add(y >> 1);
            y = y.wrapping_sub(w >> 1);
            p[0] = x; p[1] = y; p[2] = z; p[3] = w;
        }
        #[allow(clippy::many_single_char_names)]
        fn $inv(p: &mut [$int; BLOCK_SIZE]) {
            let mut x = p[0];
            let mut y = p[1];
            let mut z = p[2];
            let mut w = p[3];
            y = y.wrapping_add(w >> 1);
            w = w.wrapping_sub(y >> 1);
            y = y.wrapping_add(w); w <<= 1; w = w.wrapping_sub(y);
            z = z.wrapping_add(x); x <<= 1; x = x.wrapping_sub(z);
            y = y.wrapping_add(z); z <<= 1; z = z.wrapping_sub(y);
            w = w.wrapping_add(x); x <<= 1; x = x.wrapping_sub(w);
            p[0] = x; p[1] = y; p[2] = z; p[3] = w;
        }
    };
}

impl_lift!(fwd_lift_i32, inv_lift_i32, i32);
impl_lift!(fwd_lift_i64, inv_lift_i64, i64);

// ---------------------------------------------------------------------------
// Negabinary conversion (signed ↔ unsigned).
// ---------------------------------------------------------------------------

#[inline]
fn int2uint_i32(x: i32) -> u32 {
    ((x as u32).wrapping_add(NBMASK_U32)) ^ NBMASK_U32
}
#[inline]
fn uint2int_i32(x: u32) -> i32 {
    ((x ^ NBMASK_U32).wrapping_sub(NBMASK_U32)) as i32
}
#[inline]
fn int2uint_i64(x: i64) -> u64 {
    ((x as u64).wrapping_add(NBMASK_U64)) ^ NBMASK_U64
}
#[inline]
fn uint2int_i64(x: u64) -> i64 {
    ((x ^ NBMASK_U64).wrapping_sub(NBMASK_U64)) as i64
}

// ---------------------------------------------------------------------------
// Embedded bit-plane encoder / decoder
//
// Direct port of LLNL/zfp's `encode_few_ints` / `decode_few_ints` from
// `src/template/encode.c` and `src/template/decode.c`, specialized for
// `size <= 64`. For the 4-coefficient 1D case the generic u32 form covers
// f32 and i32 blocks; the u64 form (added for 2D/3D) covers f64 and i64.
//
// Algorithm: from most-significant bit plane down to `kmin`, each iteration:
//   * Write the first `n` bits of the plane verbatim — one refinement bit
//     per already-significant coefficient (they've had their group-test
//     "1" emitted in a previous plane and are now in fixed-precision mode).
//   * Run a unary run-length scan over the remaining `size - n`
//     coefficients. Repeatedly: emit a group-test bit. If 0, done with
//     this plane. If 1, scan one-by-one; emit each coefficient's bit at
//     this plane until a 1-bit is found (that coefficient becomes newly
//     significant). If only one coefficient remains after a positive
//     group test, its significance is implicit.
// ---------------------------------------------------------------------------

/// `encode_few_ints` / `decode_few_ints` ports. Generated per integer width.
///
/// `size` must be ≤ 64 — the bit plane is packed into a single u64 register.
/// For N-D blocks where size > 64 (4D has 256), a separate `many_ints`
/// variant will be needed; Step 5 will add it.
macro_rules! impl_few_ints {
    ($enc:ident, $dec:ident, $uint:ty, $intprec:expr) => {
        fn $enc(
            w: &mut BitWriter,
            maxbits: usize,
            maxprec: u32,
            data: &[$uint],
            size: usize,
        ) -> usize {
            let kmin: u32 = $intprec.saturating_sub(maxprec);
            let mut bits = maxbits;
            let mut n: usize = 0;
            let mut k: u32 = $intprec;
            while bits > 0 && k > kmin {
                k -= 1;
                // step 1: extract bit plane k into x (bit i = data[i] >> k & 1)
                let mut x: u64 = 0;
                for i in 0..size {
                    x |= ((data[i] >> k) as u64 & 1) << i;
                }
                // step 2: emit first n bits (refinement for already-sig coefs)
                let m = n.min(bits);
                bits -= m;
                if m > 0 {
                    w.write(m as u32, x);
                    x >>= m;
                }
                // step 3: unary run-length encode remainder
                while bits > 0 && n < size {
                    bits -= 1;
                    let group = x != 0;
                    w.write_bit(group);
                    if !group {
                        break;
                    }
                    while bits > 0 && n < size - 1 {
                        bits -= 1;
                        let bit = (x & 1) != 0;
                        w.write_bit(bit);
                        if bit {
                            break;
                        }
                        x >>= 1;
                        n += 1;
                    }
                    x >>= 1;
                    n += 1;
                }
            }
            maxbits - bits
        }

        fn $dec(
            r: &mut BitReader<'_>,
            maxbits: usize,
            maxprec: u32,
            data: &mut [$uint],
            size: usize,
        ) -> usize {
            let kmin: u32 = $intprec.saturating_sub(maxprec);
            let mut bits = maxbits;
            let mut n: usize = 0;
            for v in data.iter_mut().take(size) {
                *v = 0;
            }
            let mut k: u32 = $intprec;
            while bits > 0 && k > kmin {
                k -= 1;
                let m = n.min(bits);
                bits -= m;
                let mut x: u64 = if m > 0 { r.read(m as u32) } else { 0 };
                while bits > 0 && n < size {
                    bits -= 1;
                    if r.read_bit() {
                        while bits > 0 && n < size - 1 {
                            bits -= 1;
                            if r.read_bit() {
                                break;
                            }
                            n += 1;
                        }
                        x |= 1u64 << n;
                    } else {
                        break;
                    }
                    n += 1;
                }
                let mut xx = x;
                let mut i = 0;
                while xx != 0 {
                    data[i] |= ((xx & 1) as $uint) << k;
                    xx >>= 1;
                    i += 1;
                }
            }
            maxbits - bits
        }
    };
}

impl_few_ints!(encode_few_ints_u32, decode_few_ints_u32, u32, 32u32);
impl_few_ints!(encode_few_ints_u64, decode_few_ints_u64, u64, 64u32);

// ---------------------------------------------------------------------------
// Block-level encode / decode
// ---------------------------------------------------------------------------

/// Encode a single block of 4 f32 values into the bit stream.
///
/// Each block gets exactly `maxbits` bits of output (fixed-rate mode).
fn encode_block_f32(w: &mut BitWriter, vals: &[f32; BLOCK_SIZE], maxbits: usize) {
    let start = w.position();

    // Step 1: block-floating-point cast
    let (emax_biased, mut icoeffs) = fwd_cast_f32(vals);

    if emax_biased == 0 {
        // Empty block -- write a 0-bit header and pad
        w.write_bit(false);
        let remaining = maxbits.saturating_sub(1);
        pad_bits(w, remaining);
        return;
    }

    // Non-empty block
    w.write_bit(true);

    // Write exponent (EBITS = 8 bits).
    // `emax_biased` holds `emax + 1` where `emax` is the raw IEEE biased
    // exponent of the block's largest-magnitude value. ZFP stores this
    // `emax + 1` value directly (matches reference LLNL/zfp output).
    w.write(EBITS_F32, u64::from(emax_biased));

    // Step 2: forward lifting transform
    fwd_lift_i32(&mut icoeffs);

    // Step 3: negabinary conversion
    let mut ucoeffs = [0u32; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        ucoeffs[i] = int2uint_i32(icoeffs[i]);
    }

    // Step 4: embedded bit-plane encoding with budget
    let header_bits = 1 + EBITS_F32 as usize;
    let coeff_bits = maxbits.saturating_sub(header_bits);
    // maxprec for 1D fixed-rate: full precision of the coefficient type.
    encode_few_ints_u32(w, coeff_bits, 32, &ucoeffs, BLOCK_SIZE);

    // Pad to exactly maxbits
    let used = w.position() - start;
    let remaining = maxbits.saturating_sub(used);
    pad_bits(w, remaining);
}

/// Decode a single block of 4 f32 values from the bit stream.
fn decode_block_f32(
    r: &mut BitReader<'_>,
    maxbits: usize,
) -> Result<[f32; BLOCK_SIZE], FormatError> {
    // Read empty-block flag
    let nonempty = r.read_bit();
    if !nonempty {
        // Skip remaining bits
        let remaining = maxbits.saturating_sub(1);
        r.read(remaining.min(64) as u32);
        if remaining > 64 {
            skip_bits(r, remaining - 64);
        }
        return Ok([0.0f32; BLOCK_SIZE]);
    }

    // Read exponent — the stored value is `emax + 1` (see encode_block_f32).
    let emax_biased = r.read(EBITS_F32) as u32;

    // Decode coefficients
    let header_bits = 1 + EBITS_F32 as usize;
    let coeff_bits = maxbits.saturating_sub(header_bits);
    let mut ucoeffs = [0u32; BLOCK_SIZE];
    let bits_consumed = decode_few_ints_u32(r, coeff_bits, 32, &mut ucoeffs, BLOCK_SIZE);

    // Skip remaining padding bits to maintain block alignment.
    let remaining = coeff_bits.saturating_sub(bits_consumed);
    skip_bits(r, remaining);

    // Negabinary -> signed
    let mut icoeffs = [0i32; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        icoeffs[i] = uint2int_i32(ucoeffs[i]);
    }

    // Inverse lifting transform
    inv_lift_i32(&mut icoeffs);

    // Inverse block-floating-point cast
    Ok(inv_cast_f32(emax_biased, &icoeffs))
}

fn encode_block_f64(w: &mut BitWriter, vals: &[f64; BLOCK_SIZE], maxbits: usize) {
    let start = w.position();
    let (emax_biased, mut icoeffs) = fwd_cast_f64(vals);
    if emax_biased == 0 {
        w.write_bit(false);
        pad_bits(w, maxbits.saturating_sub(1));
        return;
    }
    w.write_bit(true);
    w.write(EBITS_F64, emax_biased);
    fwd_lift_i64(&mut icoeffs);
    let mut ucoeffs = [0u64; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        ucoeffs[i] = int2uint_i64(icoeffs[i]);
    }
    let header_bits = 1 + EBITS_F64 as usize;
    let coeff_bits = maxbits.saturating_sub(header_bits);
    encode_few_ints_u64(w, coeff_bits, 64, &ucoeffs, BLOCK_SIZE);
    let used = w.position() - start;
    pad_bits(w, maxbits.saturating_sub(used));
}

fn decode_block_f64(
    r: &mut BitReader<'_>,
    maxbits: usize,
) -> Result<[f64; BLOCK_SIZE], FormatError> {
    let nonempty = r.read_bit();
    if !nonempty {
        let remaining = maxbits.saturating_sub(1);
        r.read(remaining.min(64) as u32);
        if remaining > 64 {
            skip_bits(r, remaining - 64);
        }
        return Ok([0.0f64; BLOCK_SIZE]);
    }
    let emax_biased = r.read(EBITS_F64);
    let header_bits = 1 + EBITS_F64 as usize;
    let coeff_bits = maxbits.saturating_sub(header_bits);
    let mut ucoeffs = [0u64; BLOCK_SIZE];
    let bits_consumed = decode_few_ints_u64(r, coeff_bits, 64, &mut ucoeffs, BLOCK_SIZE);
    skip_bits(r, coeff_bits.saturating_sub(bits_consumed));
    let mut icoeffs = [0i64; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        icoeffs[i] = uint2int_i64(ucoeffs[i]);
    }
    inv_lift_i64(&mut icoeffs);
    Ok(inv_cast_f64(emax_biased, &icoeffs))
}

/// Integer-path block encoder — no block-float header, coefficients are the
/// input values cast directly. Covers i32 (u32 coeff) and i64 (u64 coeff).
macro_rules! impl_int_block_codec {
    ($enc:ident, $dec:ident, $int:ty, $uint:ty, $fwd_lift:ident, $inv_lift:ident,
     $to_uint:ident, $to_int:ident, $enc_ints:ident, $dec_ints:ident, $intprec:expr) => {
        fn $enc(w: &mut BitWriter, vals: &[$int; BLOCK_SIZE], maxbits: usize) {
            let start = w.position();
            let mut icoeffs = *vals;
            $fwd_lift(&mut icoeffs);
            let mut ucoeffs = [0 as $uint; BLOCK_SIZE];
            for i in 0..BLOCK_SIZE {
                ucoeffs[i] = $to_uint(icoeffs[i]);
            }
            $enc_ints(w, maxbits, $intprec, &ucoeffs, BLOCK_SIZE);
            let used = w.position() - start;
            pad_bits(w, maxbits.saturating_sub(used));
        }
        fn $dec(
            r: &mut BitReader<'_>,
            maxbits: usize,
        ) -> Result<[$int; BLOCK_SIZE], FormatError> {
            let mut ucoeffs = [0 as $uint; BLOCK_SIZE];
            let bits_consumed = $dec_ints(r, maxbits, $intprec, &mut ucoeffs, BLOCK_SIZE);
            skip_bits(r, maxbits.saturating_sub(bits_consumed));
            let mut icoeffs = [0 as $int; BLOCK_SIZE];
            for i in 0..BLOCK_SIZE {
                icoeffs[i] = $to_int(ucoeffs[i]);
            }
            $inv_lift(&mut icoeffs);
            Ok(icoeffs)
        }
    };
}

impl_int_block_codec!(
    encode_block_i32, decode_block_i32,
    i32, u32, fwd_lift_i32, inv_lift_i32, int2uint_i32, uint2int_i32,
    encode_few_ints_u32, decode_few_ints_u32, 32u32
);
impl_int_block_codec!(
    encode_block_i64, decode_block_i64,
    i64, u64, fwd_lift_i64, inv_lift_i64, int2uint_i64, uint2int_i64,
    encode_few_ints_u64, decode_few_ints_u64, 64u32
);

/// Write `n` zero bits.
fn pad_bits(w: &mut BitWriter, n: usize) {
    // Write in chunks of 64
    let mut remaining = n;
    while remaining >= 64 {
        w.write(64, 0);
        remaining -= 64;
    }
    if remaining > 0 {
        w.write(remaining as u32, 0);
    }
}

/// Skip `n` bits in the reader.
fn skip_bits(r: &mut BitReader<'_>, n: usize) {
    let mut remaining = n;
    while remaining >= 64 {
        r.read(64);
        remaining -= 64;
    }
    if remaining > 0 {
        r.read(remaining as u32);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// ZFP compression mode stored in cd\_values\[0\].
const ZFP_MODE_RATE: u32 = 1;

/// Compress an f32 array using ZFP fixed-rate mode.
///
/// The input is a byte slice containing little-endian f32 values. The `rate`
/// parameter specifies bits per value (e.g. 16.0 means each block of 4 values
/// gets 64 bits of compressed output).
///
/// Returns the compressed byte stream.
pub fn compress_f32(data: &[u8], rate: f64) -> Result<Vec<u8>, FormatError> {
    let num_floats = data.len() / 4;
    let maxbits = (rate * BLOCK_SIZE as f64) as usize;
    let num_blocks = num_floats.div_ceil(BLOCK_SIZE);
    let total_bits = num_blocks * maxbits;

    let mut w = BitWriter::new();

    let mut i = 0;
    while i < num_floats {
        let mut block = [0.0f32; BLOCK_SIZE];
        let n_real = (num_floats - i).min(BLOCK_SIZE);
        for j in 0..n_real {
            let off = (i + j) * 4;
            block[j] =
                f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        }
        pad_partial_block_f32(&mut block, n_real);
        encode_block_f32(&mut w, &block, maxbits);
        i += BLOCK_SIZE;
    }

    let mut out = w.finish();
    // Truncate trailing word-alignment padding so that the output length
    // matches the exact bit budget (what H5Z-ZFP stores on disk).
    out.truncate(total_bits.div_ceil(8));
    Ok(out)
}

/// Decompress a ZFP-compressed f32 stream.
///
/// `compressed` is the raw compressed bytes. `num_floats` is the number of
/// f32 values in the original uncompressed data. `rate` is the bits-per-value
/// rate used during compression.
///
/// Returns the decompressed data as little-endian f32 bytes.
pub fn decompress_f32(
    compressed: &[u8],
    num_floats: usize,
    rate: f64,
) -> Result<Vec<u8>, FormatError> {
    let maxbits = (rate * BLOCK_SIZE as f64) as usize;

    let mut r = BitReader::new(compressed);
    let mut output = Vec::with_capacity(num_floats * 4);

    let mut i = 0;
    while i < num_floats {
        let block = decode_block_f32(&mut r, maxbits)?;
        let count = BLOCK_SIZE.min(num_floats - i);
        for j in 0..count {
            output.extend_from_slice(&block[j].to_le_bytes());
        }
        i += BLOCK_SIZE;
    }

    Ok(output)
}

/// Per-type 1D compress/decompress. Same structure as `compress_f32` /
/// `decompress_f32` but parameterized over scalar width and the block
/// encoder/decoder/padder used.
macro_rules! impl_compress_1d {
    ($compress:ident, $decompress:ident, $scalar:ty, $zero:expr,
     $elem_bytes:expr, $from_le:path, $encode_block:ident, $decode_block:ident,
     $pad_partial:ident) => {
        #[doc = concat!("Compress a 1D ", stringify!($scalar), " array via ZFP fixed-rate.")]
        pub fn $compress(data: &[u8], rate: f64) -> Result<Vec<u8>, FormatError> {
            const ESZ: usize = $elem_bytes;
            let num_values = data.len() / ESZ;
            let maxbits = (rate * BLOCK_SIZE as f64) as usize;
            let num_blocks = num_values.div_ceil(BLOCK_SIZE);
            let total_bits = num_blocks * maxbits;
            let mut w = BitWriter::new();
            let mut i = 0;
            while i < num_values {
                let mut block = [$zero; BLOCK_SIZE];
                let n_real = (num_values - i).min(BLOCK_SIZE);
                for j in 0..n_real {
                    let off = (i + j) * ESZ;
                    let mut buf = [0u8; ESZ];
                    buf.copy_from_slice(&data[off..off + ESZ]);
                    block[j] = $from_le(buf);
                }
                $pad_partial(&mut block, n_real);
                $encode_block(&mut w, &block, maxbits);
                i += BLOCK_SIZE;
            }
            let mut out = w.finish();
            out.truncate(total_bits.div_ceil(8));
            Ok(out)
        }

        #[doc = concat!("Decompress to 1D ", stringify!($scalar), " bytes (little-endian).")]
        pub fn $decompress(
            compressed: &[u8],
            num_values: usize,
            rate: f64,
        ) -> Result<Vec<u8>, FormatError> {
            const ESZ: usize = $elem_bytes;
            let maxbits = (rate * BLOCK_SIZE as f64) as usize;
            let mut r = BitReader::new(compressed);
            let mut output = Vec::with_capacity(num_values * ESZ);
            let mut i = 0;
            while i < num_values {
                let block = $decode_block(&mut r, maxbits)?;
                let count = BLOCK_SIZE.min(num_values - i);
                for j in 0..count {
                    output.extend_from_slice(&block[j].to_le_bytes());
                }
                i += BLOCK_SIZE;
            }
            Ok(output)
        }
    };
}

#[inline]
fn f64_from_le(b: [u8; 8]) -> f64 {
    f64::from_le_bytes(b)
}
#[inline]
fn i32_from_le(b: [u8; 4]) -> i32 {
    i32::from_le_bytes(b)
}
#[inline]
fn i64_from_le(b: [u8; 8]) -> i64 {
    i64::from_le_bytes(b)
}

impl_compress_1d!(
    compress_f64, decompress_f64, f64, 0.0f64,
    8, f64_from_le, encode_block_f64, decode_block_f64, pad_partial_block_f64
);
impl_compress_1d!(
    compress_i32, decompress_i32, i32, 0i32,
    4, i32_from_le, encode_block_i32, decode_block_i32, pad_partial_block_i32
);
impl_compress_1d!(
    compress_i64, decompress_i64, i64, 0i64,
    8, i64_from_le, encode_block_i64, decode_block_i64, pad_partial_block_i64
);

/// Build ZFP cd\_values for HDF5 filter metadata (fixed-rate mode).
///
/// Returns a `Vec<u32>` suitable for use as `FilterDescription::client_data`.
/// Format: `[mode, rate_hi, rate_lo, type, 0, 0]` where rate is stored as
/// the two u32 halves of an f64.
pub fn zfp_cd_values_rate(rate: f64) -> Vec<u32> {
    let rate_bits = rate.to_bits();
    let rate_lo = rate_bits as u32;
    let rate_hi = (rate_bits >> 32) as u32;
    // cd_values format per H5Z-ZFP:
    // [0] = mode (1=rate)
    // [1] = high 32 bits of rate (as f64)
    // [2] = low 32 bits of rate (as f64)
    // [3] = type (0=f32 unused in our case, but kept for compat)
    // [4] = 0 (unused)
    // [5] = 0 (unused)
    vec![ZFP_MODE_RATE, rate_hi, rate_lo, 0, 0, 0]
}

/// Extract the rate from ZFP cd\_values (fixed-rate mode).
///
/// Returns `Some(rate)` if the cd\_values represent fixed-rate mode, `None`
/// otherwise.
pub fn zfp_rate_from_cd_values(cd_values: &[u32]) -> Option<f64> {
    if cd_values.len() < 3 {
        return None;
    }
    if cd_values[0] != ZFP_MODE_RATE {
        return None;
    }
    let bits = (u64::from(cd_values[1]) << 32) | u64::from(cd_values[2]);
    Some(f64::from_bits(bits))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- BitWriter / BitReader round-trip --

    #[test]
    fn bitstream_roundtrip() {
        let mut w = BitWriter::new();
        w.write(3, 0b101);
        w.write(8, 0xAB);
        w.write(1, 1);
        w.write_bit(false);
        w.write(64, 0xDEAD_BEEF_CAFE_BABE);
        let buf = w.finish();

        let mut r = BitReader::new(&buf);
        assert_eq!(r.read(3), 0b101);
        assert_eq!(r.read(8), 0xAB);
        assert_eq!(r.read(1), 1);
        assert!(!r.read_bit());
        assert_eq!(r.read(64), 0xDEAD_BEEF_CAFE_BABE);
    }

    // -- Negabinary --

    #[test]
    fn negabinary_roundtrip() {
        for x in [-1000, -1, 0, 1, 42, i32::MIN, i32::MAX] {
            assert_eq!(uint2int_i32(int2uint_i32(x)), x);
        }
    }

    // -- Lifting transform --

    #[test]
    fn lift_roundtrip() {
        let original = [100, -200, 300, -400];
        let mut p = original;
        fwd_lift_i32(&mut p);
        inv_lift_i32(&mut p);
        assert_eq!(p, original);
    }

    #[test]
    fn lift_zeros() {
        let mut p = [0, 0, 0, 0];
        fwd_lift_i32(&mut p);
        assert_eq!(p, [0, 0, 0, 0]);
        inv_lift_i32(&mut p);
        assert_eq!(p, [0, 0, 0, 0]);
    }

    // -- Block-floating-point cast --

    #[test]
    fn cast_zeros() {
        let vals = [0.0f32; 4];
        let (e, c) = fwd_cast_f32(&vals);
        assert_eq!(e, 0);
        assert_eq!(c, [0, 0, 0, 0]);
    }

    #[test]
    fn cast_roundtrip_ones() {
        let vals = [1.0f32, -1.0, 2.0, -0.5];
        let (e, c) = fwd_cast_f32(&vals);
        assert!(e > 0);
        let reconstructed = inv_cast_f32(e, &c);
        for i in 0..4 {
            let err = (vals[i] - reconstructed[i]).abs();
            assert!(
                err < 1e-6,
                "value {i}: expected {}, got {}, err={err}",
                vals[i],
                reconstructed[i]
            );
        }
    }

    // -- Full codec round-trip --

    #[test]
    fn compress_decompress_zeros() {
        let data = vec![0u8; 16]; // 4 zero f32 values
        let compressed = compress_f32(&data, 16.0).unwrap();
        let decompressed = decompress_f32(&compressed, 4, 16.0).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn compress_decompress_ones() {
        let vals: Vec<f32> = vec![1.0; 8];
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let compressed = compress_f32(&data, 16.0).unwrap();
        let decompressed = decompress_f32(&compressed, 8, 16.0).unwrap();
        // Lossy -- check within tolerance
        let recon: Vec<f32> = decompressed
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for (i, (&orig, &rec)) in vals.iter().zip(recon.iter()).enumerate() {
            let err = (orig - rec).abs();
            assert!(
                err < 0.01,
                "value {i}: expected {orig}, got {rec}, err={err}"
            );
        }
    }

    #[test]
    fn compress_decompress_varied() {
        // Values within similar magnitude ranges per block of 4 (ZFP uses
        // block-floating-point, so values spanning many orders of magnitude
        // within one block lose precision on the smaller ones).
        let vals: Vec<f32> = vec![
            1.0, 2.0, -1.5, 3.0, // block 0: similar magnitudes
            100.0, -50.5, 42.0, 80.0, // block 1: similar magnitudes
            0.001, 0.002, -0.003, 0.004, // block 2: similar magnitudes
        ];
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        // High rate should give good accuracy for same-magnitude blocks
        let compressed = compress_f32(&data, 24.0).unwrap();
        let decompressed = decompress_f32(&compressed, vals.len(), 24.0).unwrap();
        let recon: Vec<f32> = decompressed
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for (i, (&orig, &rec)) in vals.iter().zip(recon.iter()).enumerate() {
            let rel_err = if orig.abs() > 1e-10 {
                ((orig - rec) / orig).abs()
            } else {
                (orig - rec).abs()
            };
            assert!(
                rel_err < 0.01,
                "value {i}: expected {orig}, got {rec}, rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn compress_decompress_high_rate_lossless_ish() {
        // At rate=32 (maximum), should be nearly lossless for normal f32 values
        let vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let compressed = compress_f32(&data, 32.0).unwrap();
        let decompressed = decompress_f32(&compressed, 4, 32.0).unwrap();
        let recon: Vec<f32> = decompressed
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for (i, (&orig, &rec)) in vals.iter().zip(recon.iter()).enumerate() {
            let err = (orig - rec).abs();
            assert!(
                err < 1e-6,
                "value {i}: expected {orig}, got {rec}, err={err}"
            );
        }
    }

    #[test]
    fn cd_values_roundtrip() {
        let rate = 16.5;
        let cd = zfp_cd_values_rate(rate);
        let recovered = zfp_rate_from_cd_values(&cd).unwrap();
        assert!((rate - recovered).abs() < f64::EPSILON);
    }

    #[test]
    fn partial_block() {
        // 6 values = 1 full block + 1 partial block (2 values + 2 zeros)
        let vals: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let compressed = compress_f32(&data, 16.0).unwrap();
        let decompressed = decompress_f32(&compressed, 6, 16.0).unwrap();
        let recon: Vec<f32> = decompressed
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(recon.len(), 6);
        for (i, (&orig, &rec)) in vals.iter().zip(recon.iter()).enumerate() {
            let err = (orig - rec).abs();
            assert!(
                err < 1.0,
                "value {i}: expected {orig}, got {rec}, err={err}"
            );
        }
    }
}
