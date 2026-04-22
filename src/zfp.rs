//! Pure-Rust ZFP compression for f32 data (1D blocks of 4).
//!
//! Implements the ZFP fixed-rate compression algorithm matching the reference
//! C implementation (<https://github.com/LLNL/zfp>). Only the f32 codec is
//! provided because the weather simulation uses f32 throughout.
//!
//! # Algorithm per block of 4 f32 values
//!
//! 1. **Block-floating-point cast** -- find max exponent, convert to i32.
//! 2. **Forward lifting transform** -- 5-stage decorrelating transform.
//! 3. **Negabinary conversion** -- `(x + NBMASK) ^ NBMASK`.
//! 4. **Embedded bit-plane encoding** -- MSB-to-LSB with group tests.
//! 5. **Fixed-rate truncation** -- each block gets exactly `rate * 4` bits.
//!
//! The bit stream uses 64-bit words in **little-endian** byte order with bits
//! packed LSB-first within each word.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::error::FormatError;

/// Number of values per ZFP block (1D).
const BLOCK_SIZE: usize = 4;

/// Number of exponent bits for f32.
const EBITS: u32 = 8;

/// Negabinary mask for 32-bit integers.
const NBMASK: u32 = 0xAAAA_AAAA;

/// Total number of bit planes for u32 coefficients.
const PBITS: u32 = 32;

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

    /// Write `n` bits from the least-significant end of `value`.
    #[inline]
    fn write(&mut self, n: u32, value: u64) {
        debug_assert!(n <= 64);
        if n == 0 {
            return;
        }
        // Pack into current word
        self.word |= value << self.bits;
        self.bits += n;
        if self.bits >= 64 {
            self.buf.extend_from_slice(&self.word.to_le_bytes());
            self.bits -= 64;
            // Carry: the high bits that didn't fit.
            // When n == 64 and bits was 0, shift amount is 64 which would be
            // UB for a fixed-width shift, so guard with a conditional.
            self.word = if self.bits > 0 {
                value >> (n - self.bits)
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

/// Convert a block of 4 f32 values to block-floating-point i32 representation.
///
/// Returns `(exponent, coefficients)` where `exponent` is the biased exponent
/// (raw IEEE exponent + 1, so that values can be reconstructed as
/// `i32 * 2^(exponent - 30 - 127)`).
///
/// If all values are zero (or subnormal), returns `(0, [0; 4])` -- the block
/// is "empty" and only a single 0-bit is emitted by the encoder.
fn fwd_cast_f32(vals: &[f32; BLOCK_SIZE]) -> (u32, [i32; BLOCK_SIZE]) {
    // Find maximum biased exponent across the 4 values.
    let mut emax: i32 = 0;
    for &v in vals {
        let bits = v.to_bits();
        let e = ((bits >> 23) & 0xFF) as i32; // biased exponent
        if e > emax {
            emax = e;
        }
    }

    if emax == 0 {
        // All zeros / subnormals -- empty block
        return (0, [0; BLOCK_SIZE]);
    }

    // bias = emax + 1 (the value stored in the header)
    let bias = (emax + 1) as u32;

    // shift: number of bits to shift the mantissa.
    // ZFP: each value is converted via ldexp(value, -emax + (PBITS - 2))
    // which is equivalent to reinterpreting the mantissa as a (PBITS-1)-bit
    // fixed-point number with the implicit leading 1 at bit (PBITS-2)=30.
    let mut result = [0i32; BLOCK_SIZE];
    for (i, &v) in vals.iter().enumerate() {
        let bits = v.to_bits();
        let sign = (bits >> 31) & 1;
        let e = ((bits >> 23) & 0xFF) as i32;
        let m = (bits & 0x007F_FFFF) as i32;

        if e == 0 {
            // Zero or subnormal -- treat as 0
            result[i] = 0;
        } else {
            // Reconstruct with implicit 1 and shift to common exponent
            let frac = m | 0x0080_0000; // 24-bit significand with implicit 1
            let shift = emax - e;
            // Place the implicit 1 at bit 30 (PBITS-2), then shift down by
            // (emax - e) to align to the common exponent.
            // 30 - 23 = 7 extra bits of precision beyond the mantissa.
            let shifted = if shift < 7 {
                frac << (7 - shift)
            } else if shift < 31 + 7 {
                frac >> (shift - 7)
            } else {
                0
            };
            result[i] = if sign != 0 { -shifted } else { shifted };
        }
    }

    (bias, result)
}

/// Inverse block-floating-point cast: reconstruct f32 values from i32
/// coefficients and biased exponent.
///
/// The forward cast placed the implicit-1 at bit 30 (PBITS-2) relative to
/// the IEEE biased exponent `emax`. To invert:
///   value = coefficient * 2^(emax - 127 - 30)
/// where `emax` is the raw IEEE biased exponent and 127 is the f32 bias.
fn inv_cast_f32(emax_biased: u32, coeffs: &[i32; BLOCK_SIZE]) -> [f32; BLOCK_SIZE] {
    let mut result = [0.0f32; BLOCK_SIZE];
    if emax_biased == 0 {
        return result;
    }
    // emax_biased = raw_biased_exponent + 1
    let emax = emax_biased as i32 - 1;

    // The forward cast maps: value -> i32 via ldexp(value, (PBITS-2) - emax + bias)
    // i.e. coefficient = value * 2^(30 - emax + 127)
    // So: value = coefficient * 2^(emax - 127 - 30)
    let exp = emax - 127 - (PBITS as i32 - 2); // emax - 127 - 30

    let scale = pow2_f64(exp);
    for (i, &c) in coeffs.iter().enumerate() {
        if c == 0 {
            result[i] = 0.0;
            continue;
        }
        let val = (c as f64) * scale;
        result[i] = val as f32;
    }

    result
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
// Lifting transform (1D, 4 values)
// ---------------------------------------------------------------------------

/// Forward decorrelating lifting transform on 4 i32 values.
///
/// Matches the ZFP C reference `_t2(fwd_lift, Int, 1)` for 1D blocks.
#[allow(clippy::many_single_char_names)]
fn fwd_lift(p: &mut [i32; BLOCK_SIZE]) {
    let mut x = p[0];
    let mut y = p[1];
    let mut z = p[2];
    let mut w = p[3];

    // Stage 1: non-orthogonal transform
    x = x.wrapping_add(w);
    x >>= 1;
    w = w.wrapping_sub(x);

    z = z.wrapping_add(y);
    z >>= 1;
    y = y.wrapping_sub(z);

    x = x.wrapping_add(z);
    x >>= 1;
    z = z.wrapping_sub(x);

    w = w.wrapping_add(y);
    w >>= 1;
    y = y.wrapping_sub(w);

    // Stage 2: additional decorrelation
    w = w.wrapping_add(y >> 1);
    y = y.wrapping_sub(w >> 1);

    p[0] = x;
    p[1] = y;
    p[2] = z;
    p[3] = w;
}

/// Inverse decorrelating lifting transform on 4 i32 values.
#[allow(clippy::many_single_char_names)]
fn inv_lift(p: &mut [i32; BLOCK_SIZE]) {
    let mut x = p[0];
    let mut y = p[1];
    let mut z = p[2];
    let mut w = p[3];

    // Undo stage 2
    y = y.wrapping_add(w >> 1);
    w = w.wrapping_sub(y >> 1);

    // Undo stage 1 (reverse order)
    y = y.wrapping_add(w);
    w <<= 1;
    w = w.wrapping_sub(y);

    z = z.wrapping_add(x);
    x <<= 1;
    x = x.wrapping_sub(z);

    y = y.wrapping_add(z);
    z <<= 1;
    z = z.wrapping_sub(y);

    w = w.wrapping_add(x);
    x <<= 1;
    x = x.wrapping_sub(w);

    p[0] = x;
    p[1] = y;
    p[2] = z;
    p[3] = w;
}

// ---------------------------------------------------------------------------
// Negabinary conversion
// ---------------------------------------------------------------------------

/// Convert signed i32 to unsigned negabinary representation.
#[inline]
fn int2uint(x: i32) -> u32 {
    ((x as u32).wrapping_add(NBMASK)) ^ NBMASK
}

/// Convert unsigned negabinary back to signed i32.
#[inline]
fn uint2int(x: u32) -> i32 {
    ((x ^ NBMASK).wrapping_sub(NBMASK)) as i32
}

// ---------------------------------------------------------------------------
// Embedded bit-plane encoder / decoder
// ---------------------------------------------------------------------------

/// Encode 4 unsigned coefficients using ZFP's embedded bit-plane codec.
///
/// Implements the ZFP C reference encoder's nested-loop structure. At each
/// bit plane (MSB to LSB):
///
///  1. Emit refinement bits for already-significant coefficients.
///  2. Run a recursive group-test on the remaining (not-yet-significant)
///     coefficients: emit 0 if none have a 1-bit at this plane, or emit 1
///     then scan one-by-one. When a newly-significant coefficient is found,
///     recurse on the remaining unsignificant ones. If only one unsignificant
///     coefficient remains and the group test was 1, its significance is
///     implicit (no bit emitted).
///
/// Writing stops when `maxbits` bits have been emitted.
fn encode_ints(w: &mut BitWriter, ucoeffs: &[u32; BLOCK_SIZE], maxbits: usize) {
    let start = w.position();
    let mut sig = [false; BLOCK_SIZE];

    for k in (0..PBITS).rev() {
        // Phase 1: refinement bits for already-significant coefficients
        for i in 0..BLOCK_SIZE {
            if sig[i] {
                if w.position() - start >= maxbits {
                    return;
                }
                w.write_bit((ucoeffs[i] >> k) & 1 != 0);
            }
        }

        // Phase 2: significance pass (recursive group test)
        if w.position() - start >= maxbits {
            return;
        }
        encode_sig_group(w, ucoeffs, k, &mut sig, maxbits, start);
    }
}

/// Recursive group-test encoder for one bit plane.
///
/// `from` is the first coefficient index to consider; only indices `>= from`
/// that are not yet significant are tested.
fn encode_sig_group(
    w: &mut BitWriter,
    ucoeffs: &[u32; BLOCK_SIZE],
    plane: u32,
    sig: &mut [bool; BLOCK_SIZE],
    maxbits: usize,
    start: usize,
) {
    // Gather not-yet-significant indices.
    let mut unsig = [0usize; BLOCK_SIZE];
    let mut n = 0usize;
    for (i, &s) in sig.iter().enumerate() {
        if !s {
            unsig[n] = i;
            n += 1;
        }
    }
    if n == 0 {
        return;
    }

    encode_sig_slice(w, ucoeffs, plane, sig, &unsig[..n], maxbits, start);
}

/// Encode significance for a slice of unsignificant coefficient indices.
fn encode_sig_slice(
    w: &mut BitWriter,
    ucoeffs: &[u32; BLOCK_SIZE],
    plane: u32,
    sig: &mut [bool; BLOCK_SIZE],
    unsig: &[usize],
    maxbits: usize,
    start: usize,
) {
    if unsig.is_empty() {
        return;
    }
    if w.position() - start >= maxbits {
        return;
    }

    // Group test: does any coefficient in `unsig` have bit `plane` set?
    let any = unsig.iter().any(|&i| (ucoeffs[i] >> plane) & 1 != 0);

    w.write_bit(any);

    if !any {
        return;
    }

    // Scan to find the newly-significant coefficient(s).
    for (idx, &i) in unsig.iter().enumerate() {
        let remaining = unsig.len() - idx;

        if remaining == 1 {
            // Last one — implicit significance (the group test already said 1).
            sig[i] = true;
            return;
        }

        if w.position() - start >= maxbits {
            return;
        }

        let is_sig = (ucoeffs[i] >> plane) & 1 != 0;
        w.write_bit(is_sig);

        if is_sig {
            sig[i] = true;
            // Recurse on remaining unsignificant coefficients.
            encode_sig_slice(w, ucoeffs, plane, sig, &unsig[idx + 1..], maxbits, start);
            return;
        }
    }
}

/// Decode 4 unsigned coefficients from the embedded bit-plane codec.
fn decode_ints(r: &mut BitReader<'_>, maxbits: usize) -> ([u32; BLOCK_SIZE], usize) {
    let mut ucoeffs = [0u32; BLOCK_SIZE];
    let mut sig = [false; BLOCK_SIZE];
    let mut bits_read: usize = 0;

    for k in (0..PBITS).rev() {
        // Phase 1: refinement bits for significant coefficients
        for i in 0..BLOCK_SIZE {
            if sig[i] {
                if bits_read >= maxbits {
                    return (ucoeffs, bits_read);
                }
                if r.read_bit() {
                    ucoeffs[i] |= 1 << k;
                }
                bits_read += 1;
            }
        }

        // Phase 2: significance pass (recursive group test)
        if bits_read >= maxbits {
            return (ucoeffs, bits_read);
        }
        // Gather unsignificant indices.
        let mut unsig = [0usize; BLOCK_SIZE];
        let mut n = 0usize;
        for (i, &s) in sig.iter().enumerate() {
            if !s {
                unsig[n] = i;
                n += 1;
            }
        }
        bits_read = decode_sig_slice(
            r,
            &mut ucoeffs,
            &mut sig,
            k,
            &unsig[..n],
            bits_read,
            maxbits,
        );
    }

    (ucoeffs, bits_read)
}

/// Decode significance for a slice of unsignificant coefficient indices.
fn decode_sig_slice(
    r: &mut BitReader<'_>,
    ucoeffs: &mut [u32; BLOCK_SIZE],
    sig: &mut [bool; BLOCK_SIZE],
    plane: u32,
    unsig: &[usize],
    mut bits_read: usize,
    maxbits: usize,
) -> usize {
    if unsig.is_empty() || bits_read >= maxbits {
        return bits_read;
    }

    // Group test bit
    let any = r.read_bit();
    bits_read += 1;

    if !any {
        return bits_read;
    }

    // Scan for newly-significant coefficients
    for (idx, &i) in unsig.iter().enumerate() {
        let remaining = unsig.len() - idx;

        if remaining == 1 {
            // Implicit significance.
            sig[i] = true;
            ucoeffs[i] |= 1 << plane;
            return bits_read;
        }

        if bits_read >= maxbits {
            return bits_read;
        }

        let is_sig = r.read_bit();
        bits_read += 1;

        if is_sig {
            sig[i] = true;
            ucoeffs[i] |= 1 << plane;
            // Recurse on remaining unsignificant coefficients.
            return decode_sig_slice(
                r,
                ucoeffs,
                sig,
                plane,
                &unsig[idx + 1..],
                bits_read,
                maxbits,
            );
        }
    }

    bits_read
}

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

    // Write exponent (EBITS = 8 bits)
    // The stored value is emax_biased - 1 = emax (raw IEEE biased exponent)
    w.write(EBITS, u64::from(emax_biased - 1));

    // Step 2: forward lifting transform
    fwd_lift(&mut icoeffs);

    // Step 3: negabinary conversion
    let mut ucoeffs = [0u32; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        ucoeffs[i] = int2uint(icoeffs[i]);
    }

    // Step 4: embedded bit-plane encoding with budget
    let header_bits = 1 + EBITS as usize;
    let coeff_bits = maxbits.saturating_sub(header_bits);
    encode_ints(w, &ucoeffs, coeff_bits);

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

    // Read exponent
    let emax = r.read(EBITS) as u32;
    let emax_biased = emax + 1;

    // Decode coefficients
    let header_bits = 1 + EBITS as usize;
    let coeff_bits = maxbits.saturating_sub(header_bits);
    let (ucoeffs, bits_consumed) = decode_ints(r, coeff_bits);

    // Skip remaining padding bits to maintain block alignment.
    let remaining = coeff_bits.saturating_sub(bits_consumed);
    skip_bits(r, remaining);

    // Negabinary -> signed
    let mut icoeffs = [0i32; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        icoeffs[i] = uint2int(ucoeffs[i]);
    }

    // Inverse lifting transform
    inv_lift(&mut icoeffs);

    // Inverse block-floating-point cast
    Ok(inv_cast_f32(emax_biased, &icoeffs))
}

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

    let mut w = BitWriter::new();

    let mut i = 0;
    while i < num_floats {
        let mut block = [0.0f32; BLOCK_SIZE];
        for j in 0..BLOCK_SIZE {
            if i + j < num_floats {
                let off = (i + j) * 4;
                block[j] =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            }
            // else: zero-padded (already 0.0)
        }
        encode_block_f32(&mut w, &block, maxbits);
        i += BLOCK_SIZE;
    }

    Ok(w.finish())
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
            assert_eq!(uint2int(int2uint(x)), x);
        }
    }

    // -- Lifting transform --

    #[test]
    fn lift_roundtrip() {
        let original = [100, -200, 300, -400];
        let mut p = original;
        fwd_lift(&mut p);
        inv_lift(&mut p);
        assert_eq!(p, original);
    }

    #[test]
    fn lift_zeros() {
        let mut p = [0, 0, 0, 0];
        fwd_lift(&mut p);
        assert_eq!(p, [0, 0, 0, 0]);
        inv_lift(&mut p);
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
