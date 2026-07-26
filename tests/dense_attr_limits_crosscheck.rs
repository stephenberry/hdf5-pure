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

mod common;
use common::heap::{huge_object_count, managed_object_count};

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
    c_child(path, "child_reads_with_libhdf5").verdict
}

/// As [`c_reads`], plus what libhdf5 made of each attribute's contents.
struct CDetail {
    verdict: CReads,
    /// Each attribute's total data size in bytes, keyed by name.
    sizes: Vec<(String, usize)>,
    /// The wrapping sum of each `i64` attribute's values, keyed by name.
    sums: Vec<(String, i64)>,
}

impl CDetail {
    #[track_caller]
    fn size_of(&self, name: &str) -> Option<usize> {
        self.sizes.iter().find(|(n, _)| n == name).map(|(_, s)| *s)
    }

    #[track_caller]
    fn sum_of(&self, name: &str) -> Option<i64> {
        self.sums.iter().find(|(n, _)| n == name).map(|(_, s)| *s)
    }
}

/// Read `path` with libhdf5 and report what it made of every attribute.
fn c_reads_in_detail(path: &std::path::Path) -> CDetail {
    c_child(path, "child_reads_attribute_sizes")
}

/// Re-exec this binary to run `child` against `path`, and parse what it reported.
fn c_child(path: &std::path::Path, child: &str) -> CDetail {
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args([child, "--exact", "--nocapture"])
        .env(CHILD_ENV, path)
        .output()
        .expect("re-exec the test binary");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let reported = stdout
        .lines()
        .find_map(|l| l.strip_prefix("ATTRS="))
        .and_then(|n| n.parse::<usize>().ok());
    let sizes = stdout
        .lines()
        .filter_map(|l| l.strip_prefix("SIZE="))
        .filter_map(|l| l.split_once('='))
        .map(|(name, n)| (name.to_string(), n.parse::<usize>().expect("a byte count")))
        .collect();
    let sums = stdout
        .lines()
        .filter_map(|l| l.strip_prefix("SUM="))
        .filter_map(|l| l.split_once('='))
        .map(|(name, n)| (name.to_string(), n.parse::<i64>().expect("a value sum")))
        .collect();

    let verdict = match (out.status.success(), reported) {
        (true, Some(n)) => CReads::Attrs(n),
        (true, None) if stdout.contains("OPEN_FAILED") => CReads::Error,
        _ => CReads::Died,
    };
    CDetail {
        verdict,
        sizes,
        sums,
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

/// The child half of [`c_reads_with_sizes`]. Reports each attribute's datatype
/// size, and for `i64` attributes the values themselves.
///
/// The datatype is part of the attribute *message*, which for a huge object lives
/// entirely outside the managed blocks, so producing these numbers at all means
/// libhdf5 followed the heap ID through the huge-objects B-tree to the right
/// bytes. The `i64` values then prove it read the whole object and not just a
/// well-formed prefix.
#[test]
fn child_reads_attribute_sizes() {
    let Ok(path) = std::env::var(CHILD_ENV) else {
        return;
    };
    match hdf5::File::open(&path) {
        Ok(f) => {
            let names = f.attr_names().expect("attribute names");
            println!("ATTRS={}", names.len());
            for name in names {
                let attr = f.attr(&name).expect("open attribute");
                let dtype = attr.dtype().expect("attribute datatype");
                println!("SIZE={name}={}", dtype.size() * attr.size());
                // Only meaningful for the integer-valued fixtures; a type
                // mismatch is an error rather than an abort, so the string
                // fixtures simply report nothing here.
                if let Ok(values) = attr.read_raw::<i64>() {
                    let sum: i64 = values
                        .iter()
                        .copied()
                        .reduce(i64::wrapping_add)
                        .unwrap_or(0);
                    println!("SUM={name}={sum}");
                }
            }
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

/// The largest text payload still stored as a managed heap object. Probed rather
/// than assumed: the threshold applies to the serialized attribute, whose
/// overhead is internal.
fn largest_managed_payload() -> usize {
    for payload in (1..=70_000).rev() {
        let bytes = nine_attrs(payload)
            .finish()
            .expect("every size is writable");
        if huge_object_count(&bytes) == 0 {
            return payload;
        }
    }
    panic!("no payload was stored as a managed object");
}

/// Both sides of the managed/huge threshold, against the library that has to read
/// them. The interesting one is the byte *past* it: the attribute's whole message
/// moves out of the direct block the name index points into, and is reachable
/// only if the heap ID, the huge-objects B-tree and the header fields all agree.
#[test]
fn c_reads_both_sides_of_the_managed_object_threshold() {
    let payload = largest_managed_payload();
    let dir = tempdir().unwrap();

    let managed = dir.path().join("managed.h5");
    nine_attrs(payload).write(&managed).unwrap();
    let detail = c_reads_in_detail(&managed);
    assert_eq!(detail.verdict, CReads::Attrs(9));
    assert_eq!(
        detail.size_of("big"),
        Some(payload),
        "libhdf5 read back a different managed attribute than was written"
    );

    let huge = dir.path().join("huge.h5");
    nine_attrs(payload + 1).write(&huge).unwrap();
    assert_eq!(
        huge_object_count(&std::fs::read(&huge).unwrap()),
        1,
        "one byte past the threshold must actually select huge storage, or this \
         test is comparing two managed heaps"
    );
    let detail = c_reads_in_detail(&huge);
    assert_eq!(detail.verdict, CReads::Attrs(9));
    assert_eq!(
        detail.size_of("big"),
        Some(payload + 1),
        "libhdf5 could not resolve the huge object"
    );
}

/// An `i64` attribute of `len` elements counting up from `first`, so both its
/// serialized size and its contents are known exactly.
fn counting_attr(first: i64, len: usize) -> AttrValue {
    AttrValue::I64Array((first..first + len as i64).collect())
}

/// The wrapping sum [`counting_attr`] implies, which is what the child reports.
fn counting_sum(first: i64, len: usize) -> i64 {
    (first..first + len as i64)
        .reduce(i64::wrapping_add)
        .unwrap_or(0)
}

/// A heap that mixes both storage classes: the direct block is sized by the
/// managed objects alone while the name index spans all of them, so libhdf5 has
/// to follow each heap ID to a different kind of destination.
#[test]
fn c_reads_a_heap_mixing_managed_and_huge_objects() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mixed.h5");

    let mut builder = FileBuilder::new();
    for i in 0..6 {
        builder.set_attr(&format!("small{i}"), counting_attr(i as i64 * 100, 10));
    }
    for i in 0..3 {
        builder.set_attr(
            &format!("big{i}"),
            counting_attr(i as i64 * 100_000, 10_000),
        );
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(huge_object_count(&bytes), 3);
    assert_eq!(managed_object_count(&bytes), 6);

    let detail = c_reads_in_detail(&path);
    assert_eq!(detail.verdict, CReads::Attrs(9));
    for i in 0..6 {
        let name = format!("small{i}");
        assert_eq!(detail.size_of(&name), Some(10 * 8));
        assert_eq!(detail.sum_of(&name), Some(counting_sum(i as i64 * 100, 10)));
    }
    for i in 0..3 {
        let name = format!("big{i}");
        assert_eq!(detail.size_of(&name), Some(10_000 * 8));
        assert_eq!(
            detail.sum_of(&name),
            Some(counting_sum(i as i64 * 100_000, 10_000)),
            "libhdf5 read the wrong bytes for huge object {i}"
        );
    }
}

/// Many huge objects in one heap, so the huge-objects B-tree holds more than the
/// single record the simplest case would. IDs are assigned in attribute order and
/// the B-tree is searched by ID, so a heap that only works for one object fails
/// here — and distinct lengths and values mean a record matched to the wrong ID
/// reads back wrong rather than coincidentally right.
#[test]
fn c_reads_a_heap_of_many_huge_objects() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("many_huge.h5");
    let len = |i: usize| 9_000 + i * 100;

    let mut builder = FileBuilder::new();
    for i in 0..12 {
        builder.set_attr(
            &format!("h{i:02}"),
            counting_attr(i as i64 * 1_000_000, len(i)),
        );
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    assert_eq!(huge_object_count(&std::fs::read(&path).unwrap()), 12);

    let detail = c_reads_in_detail(&path);
    assert_eq!(detail.verdict, CReads::Attrs(12));
    for i in 0..12 {
        let name = format!("h{i:02}");
        assert_eq!(
            detail.size_of(&name),
            Some(len(i) * 8),
            "huge object {i} came back with the wrong length"
        );
        assert_eq!(
            detail.sum_of(&name),
            Some(counting_sum(i as i64 * 1_000_000, len(i))),
            "huge object {i} came back with the wrong contents"
        );
    }
}

/// What the managed/huge split exists to avoid. The writer never emits a heap
/// whose managed objects exceed its declared limit — it moves them to huge
/// storage instead — so this reproduces that condition by hand, writing a good
/// file and then editing the limit in its fractal-heap header down below the
/// objects already stored in it. libhdf5 must not read that file back happily;
/// on an assertion-enabled build it aborts, which is both what the split
/// prevents and proof that the child-process harness above can detect it.
#[test]
fn c_rejects_a_heap_whose_objects_exceed_its_declared_limit() {
    let dir = tempdir().unwrap();
    let good = dir.path().join("good.h5");
    nine_attrs(largest_managed_payload()).write(&good).unwrap();
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

    // And the writer never produces that shape: one byte past the threshold the
    // attribute leaves the managed blocks entirely rather than overflowing the
    // declared limit.
    let past = nine_attrs(largest_managed_payload() + 1).finish().unwrap();
    assert_eq!(huge_object_count(&past), 1);
    assert_eq!(managed_object_count(&past), 8);
}

/// The attribute-count boundary, against the library that defines it. The limit
/// is 61,680 rather than the 65,535 the B-tree leaf's record-count field would
/// suggest, because libhdf5 derives the width it needs from the leaf's capacity,
/// which follows the power-of-two node size this writer declares. Getting this
/// wrong is not a near-miss: at 61,681 the file it produced aborted libhdf5, so
/// the accepted side has to be confirmed here and not just against our own
/// reader.
#[test]
fn c_reads_the_largest_accepted_attribute_count() {
    let build = |n: usize| {
        let mut builder = FileBuilder::new();
        for i in 0..n {
            builder.set_attr(&format!("a{i:06}"), AttrValue::I64(i as i64));
        }
        builder.create_dataset("x").with_f64_data(&[1.0]);
        builder
    };

    let dir = tempdir().unwrap();
    let path = dir.path().join("count_limit.h5");
    build(61_680).write(&path).unwrap();
    assert_eq!(c_reads(&path), CReads::Attrs(61_680));

    // And one past it never reaches a file at all.
    assert!(build(61_681).finish().is_err());
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
