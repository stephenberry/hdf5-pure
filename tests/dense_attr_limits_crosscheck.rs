// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit-pointer targets; skip them on 32-bit so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(not(target_pointer_width = "32"))]
//! Reference-C-library interop for the dense (fractal-heap) attribute bounds
//! (issue #191).
//!
//! The bound this crate enforces is one the reference C library enforces too, and
//! it enforces it by *aborting*: an assertion-enabled build hits
//! `H5A__attr_release_table` and raises SIGABRT rather than returning an error.
//! A test that opened such a file in-process would take the whole test binary
//! down with it, so every read here happens in a re-exec of this binary and the
//! parent inspects the child's exit status. That also makes the abort itself an
//! observable, asserted outcome rather than a crash.

use hdf5_pure::{AttrValue, FileBuilder};
use tempfile::tempdir;

/// Set on the child process to make it open `$DENSE_XCHECK_FILE` with libhdf5
/// and report what it found on stdout.
const CHILD_ENV: &str = "DENSE_XCHECK_FILE";

/// What libhdf5 made of a file, as observed from the parent.
#[derive(Debug, PartialEq, Eq)]
enum CReads {
    /// The library opened the file and reported this many attributes.
    Attrs(usize),
    /// The library rejected the file without crashing.
    Error,
    /// The library died — an assertion abort or a signal.
    Died,
}

/// Open `path` with libhdf5 in a child process, so an abort is a status rather
/// than the end of this test run.
fn c_reads(path: &std::path::Path) -> CReads {
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args(["child_reads_with_libhdf5", "--exact", "--nocapture"])
        .env(CHILD_ENV, path)
        .output()
        .expect("re-exec the test binary");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let reported = stdout
        .lines()
        .find_map(|l| l.strip_prefix("ATTRS="))
        .and_then(|n| n.parse::<usize>().ok());

    match (out.status.success(), reported) {
        (true, Some(n)) => CReads::Attrs(n),
        (true, None) if stdout.contains("OPEN_FAILED") => CReads::Error,
        _ => CReads::Died,
    }
}

/// The child half of [`c_reads`]. Inert unless the environment variable is set,
/// so a normal `cargo test` run treats it as a trivially passing test.
#[test]
fn child_reads_with_libhdf5() {
    let Ok(path) = std::env::var(CHILD_ENV) else {
        return;
    };
    match hdf5::File::open(&path) {
        Ok(f) => {
            let n = f.attr_names().expect("attribute names").len();
            println!("ATTRS={n}");
            f.close().expect("close");
        }
        Err(_) => println!("OPEN_FAILED"),
    }
}

/// Nine attributes — enough to select dense storage — the first sized to
/// `payload` bytes of text and the rest small.
fn nine_attrs(payload: usize) -> FileBuilder {
    let mut builder = FileBuilder::new();
    builder.set_attr("big", AttrValue::AsciiString("y".repeat(payload)));
    for i in 0..8 {
        builder.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder
}

/// The largest text payload the dense path accepts. Probed rather than assumed:
/// the limit applies to the serialized attribute, whose overhead is internal.
fn largest_accepted_payload() -> usize {
    for payload in (1..=70_000).rev() {
        if nine_attrs(payload).finish().is_ok() {
            return payload;
        }
    }
    panic!("no payload was accepted");
}

/// The accepted side of the boundary: the largest attribute this writer will put
/// in dense storage must produce a heap libhdf5 reads back in full. Without this
/// the bound could sit anywhere below the real limit and still look correct.
#[test]
fn c_reads_the_largest_accepted_dense_attribute() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("at_limit.h5");
    nine_attrs(largest_accepted_payload()).write(&path).unwrap();

    assert_eq!(c_reads(&path), CReads::Attrs(9));
}

/// The refused side of the boundary. The writer will no longer emit a heap whose
/// objects exceed its declared managed-object limit, so this reproduces that
/// condition by hand — writing a good file and then editing the limit in its
/// fractal-heap header down below the objects already stored in it. libhdf5 must
/// not read that file back happily; on an assertion-enabled build it aborts,
/// which is what the writer's refusal exists to prevent, and which this also
/// proves the child-process harness above can actually detect.
#[test]
fn c_rejects_a_heap_whose_objects_exceed_its_declared_limit() {
    let dir = tempdir().unwrap();
    let good = dir.path().join("good.h5");
    nine_attrs(largest_accepted_payload()).write(&good).unwrap();
    assert_eq!(c_reads(&good), CReads::Attrs(9));

    // Fractal heap header: "FRHP", version(1), heap ID length(2), I/O filter
    // length(2), flags(1), then the 4-byte maximum managed object size.
    let mut bytes = std::fs::read(&good).unwrap();
    let frhp = bytes
        .windows(4)
        .position(|w| w == b"FRHP")
        .expect("a dense heap has a fractal heap header");
    let max_managed = frhp + 4 + 1 + 2 + 2 + 1;
    let stored = u32::from_le_bytes(bytes[max_managed..max_managed + 4].try_into().unwrap());
    assert!(stored > 1_000, "expected a real limit, found {stored}");
    bytes[max_managed..max_managed + 4].copy_from_slice(&1_000u32.to_le_bytes());

    // The checksum over the header is now wrong as well, which is itself a valid
    // reason for the library to reject the file — either way it must not report
    // the attributes as readable.
    let bad = dir.path().join("bad.h5");
    std::fs::write(&bad, &bytes).unwrap();
    assert_ne!(
        c_reads(&bad),
        CReads::Attrs(9),
        "libhdf5 accepted a heap whose objects exceed its own declared limit"
    );

    // And the writer refuses to produce that shape in the first place.
    assert!(nine_attrs(largest_accepted_payload() + 1).finish().is_err());
}

/// A multi-megabyte dense heap of individually small attributes is inside what
/// the emitter can encode, and libhdf5 must accept it — including the maximum
/// direct block size the header now declares to match its own root block.
#[test]
fn c_reads_a_multi_megabyte_dense_heap() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("big_total.h5");

    let mut builder = FileBuilder::new();
    for i in 0..40 {
        builder.set_attr(&format!("a{i}"), AttrValue::AsciiString("y".repeat(60_000)));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    assert!(std::fs::metadata(&path).unwrap().len() > 2_000_000);
    assert_eq!(c_reads(&path), CReads::Attrs(40));
}
