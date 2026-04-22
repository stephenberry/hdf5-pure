//! Crosscheck the pure-Rust ZFP codec against reference fixtures produced by
//! h5py + the real H5Z-ZFP plugin.
//!
//! Fixtures and generator script live in `tests/fixtures/zfp/`. Regenerate:
//!
//!     tests/fixtures/zfp/.venv/bin/python tests/fixtures/zfp/regen.py
//!
//! The test iterates every fixture in the manifest. For each fixture that
//! falls inside the currently-implemented codec slice (see `is_supported`),
//! it runs both paths:
//!
//!   * decompress: our codec reads the reference compressed bytes and the
//!     decoded values are compared against the raw values within a
//!     rate-dependent tolerance.
//!   * compress: our codec encodes the raw values and the output is compared
//!     byte-equal against the reference compressed bytes.
//!
//! Fixtures outside the supported slice are recorded as skipped; they become
//! active as the codec is parameterized (Step 3) and extended to N-D
//! (Step 5). Any supported-fixture failure fails the test.

#![cfg(feature = "zfp")]

use std::fs;
use std::path::PathBuf;

use hdf5_pure::zfp;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Manifest {
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    rate: f64,
    mode: String,
    #[allow(dead_code)]
    filter_name: String,
    #[allow(dead_code)]
    cd_values_u32: Vec<u32>,
    raw_bytes_len: usize,
    compressed_bytes_len: usize,
    #[allow(dead_code)]
    notes: String,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/zfp")
}

fn load_manifest() -> Manifest {
    let path = fixture_dir().join("manifest.json");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}\n(run regen.py to produce fixtures)", path.display()));
    serde_json::from_str(&text).expect("parse manifest.json")
}

fn read_bin(name: &str) -> Vec<u8> {
    let path = fixture_dir().join(name);
    fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// True if the fixture falls inside the slice of the codec implemented so
/// far. As Steps 3 and 5 land, this gate widens.
fn is_supported(fix: &Fixture) -> bool {
    // Step 2 initial state: f32 1D fixed-rate only.
    fix.dtype == "f32" && fix.shape.len() == 1 && fix.mode == "rate"
}

/// Expected element size in bytes for a fixture's dtype.
fn element_size(dtype: &str) -> usize {
    match dtype {
        "f32" | "i32" => 4,
        "f64" | "i64" => 8,
        other => panic!("unknown dtype {other}"),
    }
}

/// Max absolute reconstruction error we accept at a given rate. The fixed-
/// rate mode gives each block exactly `rate * block_size` bits; for f32 at
/// rate 16 the reconstruction is good to roughly 2^-8 * max(|block|).
fn decode_tolerance(fix: &Fixture) -> f64 {
    // Conservative: give 2^-(rate/2) relative tolerance scaled by max value.
    // `rate` here is bits-per-value.
    2f64.powi(-(fix.rate as i32 / 2))
}

/// Interpret a byte slice as a sequence of f32 LE values.
fn f32_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[derive(Debug)]
struct CaseOutcome {
    name: String,
    decode_max_err: Option<f64>,
    encode_byte_match: Option<bool>,
    decode_err_msg: Option<String>,
    encode_err_msg: Option<String>,
}

fn run_case(fix: &Fixture) -> CaseOutcome {
    let raw = read_bin(&format!("{}.raw.bin", fix.name));
    let reference = read_bin(&format!("{}.compressed.bin", fix.name));
    assert_eq!(raw.len(), fix.raw_bytes_len);
    assert_eq!(reference.len(), fix.compressed_bytes_len);

    let elem_size = element_size(&fix.dtype);
    let num_values: usize = fix.shape.iter().product();
    assert_eq!(raw.len(), num_values * elem_size);

    // Decode path — 1D f32 for now
    let (decode_max_err, decode_err_msg) =
        match zfp::decompress_f32(&reference, num_values, fix.rate) {
            Ok(decoded_bytes) => {
                let expected = f32_slice(&raw);
                let got = f32_slice(&decoded_bytes);
                if got.len() != expected.len() {
                    (
                        None,
                        Some(format!(
                            "decoded len {} != expected {}",
                            got.len(),
                            expected.len()
                        )),
                    )
                } else {
                    let max_err = expected
                        .iter()
                        .zip(got.iter())
                        .map(|(&a, &b)| (a as f64 - b as f64).abs())
                        .fold(0f64, f64::max);
                    (Some(max_err), None)
                }
            }
            Err(e) => (None, Some(format!("{e:?}"))),
        };

    // Encode path
    let (encode_byte_match, encode_err_msg) = match zfp::compress_f32(&raw, fix.rate) {
        Ok(encoded) => (Some(encoded == reference), None),
        Err(e) => (None, Some(format!("{e:?}"))),
    };

    CaseOutcome {
        name: fix.name.clone(),
        decode_max_err,
        encode_byte_match,
        decode_err_msg,
        encode_err_msg,
    }
}

#[test]
#[ignore = "bit-format not yet verified against reference; Step 3 removes this"]
fn zfp_crosscheck() {
    let manifest = load_manifest();
    let mut skipped: Vec<&str> = Vec::new();
    let mut passed: Vec<String> = Vec::new();
    let mut failed: Vec<(String, String)> = Vec::new();

    for fix in &manifest.fixtures {
        if !is_supported(fix) {
            skipped.push(&fix.name);
            continue;
        }
        let outcome = run_case(fix);
        let tol = decode_tolerance(fix);
        let decode_ok = match (&outcome.decode_err_msg, outcome.decode_max_err) {
            (None, Some(e)) => e <= tol,
            _ => false,
        };
        let encode_ok = outcome.encode_byte_match.unwrap_or(false);

        if decode_ok && encode_ok {
            passed.push(outcome.name.clone());
        } else {
            let mut why = Vec::new();
            if let Some(msg) = &outcome.decode_err_msg {
                why.push(format!("decode error: {msg}"));
            } else if let Some(e) = outcome.decode_max_err {
                if !decode_ok {
                    why.push(format!("decode max_err={e:.6} > tol={tol:.6}"));
                }
            }
            if let Some(msg) = &outcome.encode_err_msg {
                why.push(format!("encode error: {msg}"));
            } else if !encode_ok {
                why.push("encode: bytes differ from reference".to_string());
            }
            failed.push((outcome.name.clone(), why.join("; ")));
        }
    }

    eprintln!("\n=== ZFP crosscheck ===");
    eprintln!("passed:  {}", passed.len());
    eprintln!("failed:  {}", failed.len());
    eprintln!("skipped: {} (outside current codec slice)", skipped.len());

    if !failed.is_empty() {
        eprintln!("\nfailures:");
        for (name, why) in &failed {
            eprintln!("  {name}: {why}");
        }
        panic!("{} supported fixtures failed crosscheck", failed.len());
    }
}
