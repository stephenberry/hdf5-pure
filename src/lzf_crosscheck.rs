//! Crosscheck the pure-Rust LZF codec against fixtures produced by h5py's
//! built-in LZF filter.
//!
//! Fixtures and generator script live in `tests/fixtures/lzf/`. Regenerate:
//!
//!     tests/fixtures/lzf/.venv/bin/python tests/fixtures/lzf/regen.py
//!
//! For every single-filter LZF fixture with an extracted chunk stream:
//!
//!   * an unmasked chunk decodes through our codec to exactly the raw bytes
//!     (LZF is lossless), and the recorded cd_values equal what our writer
//!     emits;
//!   * a masked chunk (h5py registers LZF as optional and stores an
//!     incompressible chunk raw) is byte-equal to the raw data;
//!   * our compressor's output for the raw bytes round-trips through our
//!     decoder. Byte-equality with liblzf's stream is deliberately not
//!     asserted — any conforming stream is valid (see the module doc in
//!     `lzf.rs`).
//!
//! Chained (shuffle+lzf) and multi-chunk fixtures are covered end-to-end in
//! `tests/lzf_roundtrip.rs`, which opens their `.h5` files through the full
//! reader.

use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

use crate::filter_pipeline::FILTER_LZF;
use crate::lzf;

#[derive(Debug, Deserialize)]
struct Manifest {
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    name: String,
    filters: Vec<u16>,
    cd_values_u32: Vec<u32>,
    /// Measured per-chunk mask; absent for multi-chunk fixtures, whose
    /// manifest records only whole-dataset facts.
    filter_mask: Option<u32>,
    raw_bytes_len: usize,
    compressed_bytes_len: Option<usize>,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/lzf")
}

fn load_manifest() -> Manifest {
    let path = fixture_dir().join("manifest.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "read {}: {e}\n(run regen.py to produce fixtures)",
            path.display()
        )
    });
    serde_json::from_str(&text).expect("parse manifest.json")
}

fn read_bin(name: &str) -> Vec<u8> {
    let path = fixture_dir().join(name);
    fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
fn lzf_crosscheck() {
    let manifest = load_manifest();
    let mut verified = 0usize;

    for fix in &manifest.fixtures {
        // Only single-filter LZF fixtures with an extracted single-chunk
        // stream are codec-comparable here; the rest are exercised through
        // the full reader in tests/lzf_roundtrip.rs.
        if fix.filters != [FILTER_LZF] || fix.compressed_bytes_len.is_none() {
            continue;
        }
        let raw = read_bin(&format!("{}.raw.bin", fix.name));
        let stored = read_bin(&format!("{}.compressed.bin", fix.name));
        assert_eq!(raw.len(), fix.raw_bytes_len, "{}: manifest drift", fix.name);

        if fix.filter_mask.unwrap_or(0) & 1 == 1 {
            // h5py skipped its optional LZF: the chunk is stored raw.
            assert_eq!(stored, raw, "{}: masked chunk must be raw", fix.name);
        } else {
            let decoded = lzf::decompress(&stored, Some(raw.len()))
                .unwrap_or_else(|e| panic!("{}: reference stream decode: {e:?}", fix.name));
            assert_eq!(decoded, raw, "{}: decode mismatch", fix.name);
        }

        // h5py's recorded cd_values are exactly what our writer emits for the
        // same chunk (a single chunk of raw_len one-byte elements).
        let expected_cd = lzf::h5py_cd_values(1, &[raw.len() as u64]);
        assert_eq!(fix.cd_values_u32, expected_cd, "{}: cd_values", fix.name);

        // Our stream need not equal liblzf's, but it must round-trip.
        let ours = lzf::compress(&raw);
        assert_eq!(
            lzf::decompress(&ours, Some(raw.len())).unwrap(),
            raw,
            "{}: our stream fails to round-trip",
            fix.name
        );
        verified += 1;
    }

    // A green result with zero positively-verified fixtures is not a real
    // pass; make the "nothing was checked" outcome observable.
    if verified == 0 {
        eprintln!(
            "WARNING: LZF crosscheck verified 0 fixtures ({} in manifest). \
             Nothing was cross-checked against h5py; run regen.py.",
            manifest.fixtures.len()
        );
    }
}
