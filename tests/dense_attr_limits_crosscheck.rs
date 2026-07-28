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

/// Comma-separated attribute names for the child to open individually, rather
/// than by iterating everything the object carries.
const CHILD_LOOKUP_ENV: &str = "DENSE_XCHECK_LOOKUP";

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
    c_child_looking_up(path, child, "")
}

/// As [`c_child`], additionally handing the child a comma-separated list of
/// attribute names to open individually.
fn c_child_looking_up(path: &std::path::Path, child: &str, lookup: &str) -> CDetail {
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args([child, "--exact", "--nocapture"])
        .env(CHILD_ENV, path)
        .env(CHILD_LOOKUP_ENV, lookup)
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

/// The child half of [`c_looks_up_by_name`]. Opens only the named attributes,
/// so libhdf5 has to *search* the name index rather than walk it.
///
/// A search descends the B-tree comparing name hashes, which is the one thing
/// iteration never checks: a tree whose records are in the wrong order still
/// yields every record in an in-order walk, and still fails every lookup that
/// takes a wrong branch.
#[test]
fn child_looks_up_attributes_by_name() {
    let Ok(path) = std::env::var(CHILD_ENV) else {
        return;
    };
    let wanted = std::env::var(CHILD_LOOKUP_ENV).unwrap_or_default();
    match hdf5::File::open(&path) {
        Ok(f) => {
            let mut found = 0usize;
            for name in wanted.split(',').filter(|n| !n.is_empty()) {
                let Ok(attr) = f.attr(name) else {
                    println!("MISSING={name}");
                    continue;
                };
                found += 1;
                let values = attr.read_raw::<i64>().expect("an i64 attribute");
                let sum: i64 = values
                    .iter()
                    .copied()
                    .reduce(i64::wrapping_add)
                    .unwrap_or(0);
                println!("SUM={name}={sum}");
            }
            println!("ATTRS={found}");
            f.close().expect("close");
        }
        Err(_) => println!("OPEN_FAILED"),
    }
}

/// Have libhdf5 open each of `names` by name and report the values it read.
fn c_looks_up_by_name(path: &std::path::Path, names: &[String]) -> CDetail {
    c_child_looking_up(path, "child_looks_up_attributes_by_name", &names.join(","))
}

/// The child half of [`c_inserts_then_reads`]. Opens the file read-write with
/// libhdf5, adds an attribute large enough to need huge storage, then reopens and
/// reports what is there.
///
/// Reading alone never consults the heap header's `next_huge_object_id`; only an
/// insert does, because that is where the C library derives the next ID from it.
/// A heap that declares the wrong value reads back perfectly and then collides on
/// the first write.
#[test]
fn child_inserts_with_libhdf5() {
    let Ok(path) = std::env::var(CHILD_ENV) else {
        return;
    };
    let added: Vec<i64> = (0..12_000).collect();
    {
        let f = match hdf5::File::open_rw(&path) {
            Ok(f) => f,
            Err(_) => {
                println!("OPEN_FAILED");
                return;
            }
        };
        f.new_attr::<i64>()
            .shape([added.len()])
            .create("inserted")
            .expect("create attribute")
            .write(&added)
            .expect("write attribute");
        f.close().expect("close");
    }
    match hdf5::File::open(&path) {
        Ok(f) => {
            let names = f.attr_names().expect("attribute names");
            println!("ATTRS={}", names.len());
            for name in names {
                let attr = f.attr(&name).expect("open attribute");
                let dtype = attr.dtype().expect("attribute datatype");
                println!("SIZE={name}={}", dtype.size() * attr.size());
            }
            f.close().expect("close");
        }
        Err(_) => println!("OPEN_FAILED"),
    }
}

/// Have libhdf5 add a huge attribute to a heap this crate wrote, then read the
/// result back.
fn c_inserts_then_reads(path: &std::path::Path) -> CDetail {
    c_child(path, "child_inserts_with_libhdf5")
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

/// A single large attribute, with no others to push the object over the compact
/// count. This is the shape the capability exists for, and it reaches dense
/// storage only because one oversized attribute selects it on its own.
///
/// Worth crosschecking separately rather than assuming it follows from the
/// nine-attribute case: an object carrying dense storage for *one* attribute is
/// well below the count at which the C library would have converted, and nothing
/// but the library itself settles whether it minds.
#[test]
fn c_reads_a_lone_huge_attribute() {
    let dir = tempdir().unwrap();
    let len = 12_000usize;

    for others in [0usize, 2] {
        let path = dir.path().join(format!("lone{others}.h5"));
        let mut builder = FileBuilder::new();
        builder.set_attr("big", counting_attr(0, len));
        for i in 0..others {
            builder.set_attr(&format!("a{i}"), AttrValue::I64(i as i64));
        }
        builder.create_dataset("x").with_f64_data(&[1.0]);
        builder.write(&path).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(huge_object_count(&bytes), 1);
        assert_eq!(managed_object_count(&bytes), others as u64);

        let detail = c_reads_in_detail(&path);
        assert_eq!(detail.verdict, CReads::Attrs(others + 1));
        assert_eq!(detail.size_of("big"), Some(len * 8));
        assert_eq!(detail.sum_of("big"), Some(counting_sum(0, len)));
    }
}

/// The heap header's `next_huge_object_id` holds the *last* ID assigned, not the
/// next one free, because the C library pre-increments it. Nothing that only
/// reads a file can tell the two apart — libhdf5 resolves a huge object through
/// the B-tree and never consults the counter. It consults it when *inserting*, so
/// an off-by-one there produces a file that reads back perfectly and then
/// collides ("record is already in B-tree") the first time anything adds to it.
///
/// This is the only test that exercises that field, and it also covers the header
/// accounting an insert has to build on: `free_space`, `managed_space_in_heap`
/// and the two object counts.
///
/// It pins the direction that matters rather than the exact value. A counter
/// below the last ID assigned collides, and this fails; one above merely skips an
/// ID, which the C library tolerates, so this would not catch it.
#[test]
fn c_adds_a_huge_attribute_to_a_heap_this_crate_wrote() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("insert.h5");
    let len = 12_000usize;

    let mut builder = FileBuilder::new();
    builder.set_attr("big", counting_attr(0, len));
    for i in 0..8 {
        builder.set_attr(&format!("a{i}"), AttrValue::I64(i as i64));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();
    assert_eq!(huge_object_count(&std::fs::read(&path).unwrap()), 1);

    let detail = c_inserts_then_reads(&path);
    assert_eq!(
        detail.verdict,
        CReads::Attrs(10),
        "libhdf5 could not add a huge attribute to this heap"
    );
    assert_eq!(detail.size_of("big"), Some(len * 8));
    assert_eq!(detail.size_of("inserted"), Some(len * 8));
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

/// Attribute counts that need a multi-level name index, against the library
/// that defines the shape. 61,680 was the ceiling while the index was a single
/// leaf grown to fit — one more produced a file that *aborted* libhdf5 rather
/// than being rejected by it, which is exactly why the accepted side has to be
/// confirmed here and not just against our own reader.
///
/// Each count crosses a different level boundary of a 512-byte-node tree: 570
/// is the first that needs depth 2, 10,260 the first that needs depth 3, and
/// 70,000 is well past the old single-leaf limit.
#[test]
fn c_reads_attribute_counts_that_need_a_multi_level_index() {
    let build = |n: usize| {
        let mut builder = FileBuilder::new();
        for i in 0..n {
            builder.set_attr(&format!("a{i:06}"), AttrValue::I64(i as i64));
        }
        builder.create_dataset("x").with_f64_data(&[1.0]);
        builder
    };

    let dir = tempdir().unwrap();
    for count in [570, 10_260, 61_681, 70_000] {
        let path = dir.path().join(format!("count_{count}.h5"));
        build(count).write(&path).unwrap();
        assert_eq!(
            c_reads(&path),
            CReads::Attrs(count),
            "libhdf5 could not read {count} attributes"
        );

        // Iteration walks the tree in order and would be satisfied by a tree
        // whose records sit in the wrong order; opening by name makes libhdf5
        // descend it, comparing at each internal node. Sampled across the range
        // so the descents take different paths.
        let sampled: Vec<String> = [0, 1, count / 3, count / 2, count - 2, count - 1]
            .iter()
            .map(|i| format!("a{i:06}"))
            .collect();
        let detail = c_looks_up_by_name(&path, &sampled);
        assert_eq!(
            detail.verdict,
            CReads::Attrs(sampled.len()),
            "libhdf5 could not look up attributes by name among {count}"
        );
        for (i, name) in [0, 1, count / 3, count / 2, count - 2, count - 1]
            .iter()
            .zip(&sampled)
        {
            assert_eq!(
                detail.sum_of(name),
                Some(*i as i64),
                "{name} read back the wrong value among {count} attributes"
            );
        }
    }
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
