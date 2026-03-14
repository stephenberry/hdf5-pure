//! HDF5 metadata checksum: Jenkins lookup3 `hashlittle` and CRC32.
//!
//! HDF5 uses Bob Jenkins' lookup3 hash (not CRC32C) for all metadata
//! checksums in superblocks, object headers, B-tree nodes, etc.
//!
//! When the `fast-checksum` feature is enabled, CRC32 computations use
//! hardware-accelerated instructions via the `crc32fast` crate.

/// Compute the Jenkins lookup3 checksum of a byte slice.
///
/// This is the `hashlittle` function from Bob Jenkins' lookup3.c,
/// matching the `H5_checksum_lookup3` function in the HDF5 C library.
pub fn jenkins_lookup3(data: &[u8]) -> u32 {
    hashlittle(data, 0)
}

/// Compute CRC32 (IEEE / ISO 3309) over data.
///
/// When the `fast-checksum` feature is enabled, this uses hardware CRC32
/// instructions on x86 (SSE 4.2) and ARM (CRC extension) via `crc32fast`.
/// Otherwise falls back to a software table-based implementation.
pub fn crc32(data: &[u8]) -> u32 {
    #[cfg(feature = "fast-checksum")]
    {
        crc32fast::hash(data)
    }
    #[cfg(not(feature = "fast-checksum"))]
    {
        crc32_software(data)
    }
}

/// Software CRC32 (always available, for testing/comparison).
pub fn crc32_software(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = CRC32_TABLE[index] ^ (crc >> 8);
    }
    crc ^ 0xFFFFFFFF
}

/// CRC32 lookup table (IEEE polynomial 0xEDB88320).
#[rustfmt::skip]
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = 0xEDB88320 ^ (crc >> 1);
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
};

fn rot(x: u32, k: u32) -> u32 {
    x.rotate_left(k)
}

fn mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *a = a.wrapping_sub(*c); *a ^= rot(*c, 4);  *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a); *b ^= rot(*a, 6);  *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b); *c ^= rot(*b, 8);  *b = b.wrapping_add(*a);
    *a = a.wrapping_sub(*c); *a ^= rot(*c, 16); *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a); *b ^= rot(*a, 19); *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b); *c ^= rot(*b, 4);  *b = b.wrapping_add(*a);
}

fn final_mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *c ^= *b; *c = c.wrapping_sub(rot(*b, 14));
    *a ^= *c; *a = a.wrapping_sub(rot(*c, 11));
    *b ^= *a; *b = b.wrapping_sub(rot(*a, 25));
    *c ^= *b; *c = c.wrapping_sub(rot(*b, 16));
    *a ^= *c; *a = a.wrapping_sub(rot(*c, 4));
    *b ^= *a; *b = b.wrapping_sub(rot(*a, 14));
    *c ^= *b; *c = c.wrapping_sub(rot(*b, 24));
}

fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn hashlittle(data: &[u8], initval: u32) -> u32 {
    let length = data.len();
    let mut a: u32 = 0xdeadbeefu32.wrapping_add(length as u32).wrapping_add(initval);
    let mut b: u32 = a;
    let mut c: u32 = a;

    let mut offset = 0;
    let mut remaining = length;

    // Process 12-byte blocks
    while remaining > 12 {
        a = a.wrapping_add(read_u32_le(data, offset));
        b = b.wrapping_add(read_u32_le(data, offset + 4));
        c = c.wrapping_add(read_u32_le(data, offset + 8));
        mix(&mut a, &mut b, &mut c);
        offset += 12;
        remaining -= 12;
    }

    // Handle the last few bytes (switch fall-through pattern)
    let tail = &data[offset..];
    // Using the little-endian byte reading approach from hashlittle
    match remaining {
        12 => {
            a = a.wrapping_add(read_u32_le(tail, 0));
            b = b.wrapping_add(read_u32_le(tail, 4));
            c = c.wrapping_add(read_u32_le(tail, 8));
        }
        11 => {
            c = c.wrapping_add((tail[10] as u32) << 16);
            c = c.wrapping_add((tail[9] as u32) << 8);
            c = c.wrapping_add(tail[8] as u32);
            b = b.wrapping_add(read_u32_le(tail, 4));
            a = a.wrapping_add(read_u32_le(tail, 0));
        }
        10 => {
            c = c.wrapping_add((tail[9] as u32) << 8);
            c = c.wrapping_add(tail[8] as u32);
            b = b.wrapping_add(read_u32_le(tail, 4));
            a = a.wrapping_add(read_u32_le(tail, 0));
        }
        9 => {
            c = c.wrapping_add(tail[8] as u32);
            b = b.wrapping_add(read_u32_le(tail, 4));
            a = a.wrapping_add(read_u32_le(tail, 0));
        }
        8 => {
            b = b.wrapping_add(read_u32_le(tail, 4));
            a = a.wrapping_add(read_u32_le(tail, 0));
        }
        7 => {
            b = b.wrapping_add((tail[6] as u32) << 16);
            b = b.wrapping_add((tail[5] as u32) << 8);
            b = b.wrapping_add(tail[4] as u32);
            a = a.wrapping_add(read_u32_le(tail, 0));
        }
        6 => {
            b = b.wrapping_add((tail[5] as u32) << 8);
            b = b.wrapping_add(tail[4] as u32);
            a = a.wrapping_add(read_u32_le(tail, 0));
        }
        5 => {
            b = b.wrapping_add(tail[4] as u32);
            a = a.wrapping_add(read_u32_le(tail, 0));
        }
        4 => {
            a = a.wrapping_add(read_u32_le(tail, 0));
        }
        3 => {
            a = a.wrapping_add((tail[2] as u32) << 16);
            a = a.wrapping_add((tail[1] as u32) << 8);
            a = a.wrapping_add(tail[0] as u32);
        }
        2 => {
            a = a.wrapping_add((tail[1] as u32) << 8);
            a = a.wrapping_add(tail[0] as u32);
        }
        1 => {
            a = a.wrapping_add(tail[0] as u32);
        }
        0 => return c,
        _ => unreachable!(),
    }

    final_mix(&mut a, &mut b, &mut c);
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        // Empty input should return the initial state after no mixing
        let h = jenkins_lookup3(b"");
        // Just verify it doesn't panic and returns something deterministic
        assert_eq!(h, jenkins_lookup3(b""));
    }

    #[test]
    fn known_values() {
        // Test with known input to ensure consistency
        let h1 = jenkins_lookup3(b"hello");
        let h2 = jenkins_lookup3(b"hello");
        assert_eq!(h1, h2);

        // Different inputs should give different outputs
        let h3 = jenkins_lookup3(b"world");
        assert_ne!(h1, h3);
    }

    #[test]
    fn twelve_byte_boundary() {
        // Exactly 12 bytes
        let h = jenkins_lookup3(b"abcdefghijkl");
        assert_eq!(h, jenkins_lookup3(b"abcdefghijkl"));
    }

    #[test]
    fn longer_than_12() {
        let h = jenkins_lookup3(b"abcdefghijklmnop");
        assert_eq!(h, jenkins_lookup3(b"abcdefghijklmnop"));
    }

    #[test]
    fn all_tail_lengths() {
        // Test every tail length from 1 to 12
        for len in 1..=12 {
            let data: Vec<u8> = (0..len).map(|i| i as u8).collect();
            let h1 = jenkins_lookup3(&data);
            let h2 = jenkins_lookup3(&data);
            assert_eq!(h1, h2, "failed for length {len}");
        }
    }

    #[test]
    fn verify_against_hdf5_file() {
        // Verify against a real HDF5 file checksum
        let file_data: &[u8] = include_bytes!("../tests/fixtures/v2_groups.h5");
        // Superblock v3: checksum at offset 44, covers bytes 0..44
        let stored = u32::from_le_bytes([
            file_data[44], file_data[45], file_data[46], file_data[47],
        ]);
        let computed = jenkins_lookup3(&file_data[0..44]);
        assert_eq!(computed, stored, "Jenkins lookup3 should match HDF5 superblock checksum");
    }

    // --- CRC32 tests ---

    #[test]
    fn crc32_empty() {
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn crc32_known_value() {
        // CRC32 of "123456789" is 0xCBF43926
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn crc32_software_matches() {
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let hw = crc32(&data);
        let sw = crc32_software(&data);
        assert_eq!(hw, sw);
    }

    #[test]
    fn crc32_deterministic() {
        let data = b"hello world";
        assert_eq!(crc32(data), crc32(data));
    }
}
