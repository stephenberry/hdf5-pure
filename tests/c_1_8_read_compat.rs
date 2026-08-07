//! Reading files written by an actual HDF5 1.8 library.
//!
//! The rest of the suite reaches old formats two ways, and both have a blind
//! spot this covers.
//!
//! `hdf5-metno` builds a *current* libhdf5, so a crosscheck can ask it for an
//! older format with `H5Pset_libver_bounds` — but a modern library writing an
//! old format is not the same artifact as an old library writing it, and every
//! such test is gated off 32-bit targets (`hdf5-metno` needs 64-bit pointers),
//! which is where the address-arithmetic bugs live. The committed fixtures under
//! `tests/fixtures/` cover superblock 0 and 3 and nothing between.
//!
//! These two files are written by HDF5 1.8.23 and committed, so they need no
//! dev-dependency and run everywhere `cargo test` does. They fill the gap at
//! superblock 1 and 2:
//!
//! - **`v1_superblock.h5`** — a version 1 superblock, which the C library writes
//!   only when a B-tree K value is non-default. Those K values are the fields
//!   the version 1 layout adds, and reading them from the wrong offsets was a
//!   real defect (fixed in 0.33.0). `src/superblock.rs` asserts the parsed
//!   fields; this asserts the file is *usable* through the public API.
//! - **`v2_superblock.h5`** — 1.8's newest format, and the one this crate now
//!   writes by default for `.mat` files. A `.mat` a C-based tool has touched
//!   comes back looking like this.
//!
//! Both hold the same content, so a difference between them is the format.
//! See `tests/fixtures/c_1_8/NOTICE.md` for how they were produced.

use hdf5_pure::{AttrValue, File};

const V1: &str = "tests/fixtures/c_1_8/v1_superblock.h5";
const V2: &str = "tests/fixtures/c_1_8/v2_superblock.h5";

/// The values `regen.c` wrote, asserted against every reader entry point.
fn assert_contents(path: &str) {
    let f = File::open(path).unwrap_or_else(|e| panic!("{path}: {e:?}"));

    // Contiguous dataset with an attribute.
    assert_eq!(
        f.dataset("values").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5, 3.5, 4.5],
        "{path}: /values"
    );
    assert_eq!(
        f.dataset("values").unwrap().attrs().unwrap().get("units"),
        Some(&AttrValue::AsciiString("m/s".into())),
        "{path}: /values units"
    );

    // Chunked and deflated. Under a pre-1.10 superblock this is a version 1
    // B-tree chunk index — the index this crate reads and does not write, so
    // nothing it produces can stand in for this file.
    let expected: Vec<i32> = (0..1000).map(|i| i % 97).collect();
    assert_eq!(
        f.dataset("chunked").unwrap().read_i32().unwrap(),
        expected,
        "{path}: /chunked"
    );

    // A group, its attribute, and a dataset inside it. On the version 1
    // superblock the group is a v1 symbol table; on the version 2 it is not.
    assert_eq!(
        f.group("grp").unwrap().attrs().unwrap().get("tag"),
        Some(&AttrValue::AsciiString("group".into())),
        "{path}: /grp tag"
    );
    assert_eq!(
        f.dataset("grp/inner").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5, 3.5, 4.5],
        "{path}: /grp/inner"
    );

    // A root-group attribute.
    assert_eq!(
        f.root().attrs().unwrap().get("root_attr"),
        Some(&AttrValue::AsciiString("r".into())),
        "{path}: / root_attr"
    );
}

#[test]
fn reads_a_c_written_version_1_superblock() {
    assert_contents(V1);
    assert_eq!(
        superblock_version(V1),
        1,
        "the fixture stopped being a version 1 superblock"
    );
}

#[test]
fn reads_a_c_written_version_2_superblock() {
    assert_contents(V2);
    assert_eq!(
        superblock_version(V2),
        2,
        "the fixture stopped being a version 2 superblock"
    );
}

/// The two formats carry the same content, so every value read from one must
/// equal the value read from the other.
///
/// Asserted because the pair is what makes either file diagnostic: a failure in
/// only one of them names the format, where a failure in both names the reader.
#[test]
fn the_two_formats_carry_the_same_content() {
    let (a, b) = (File::open(V1).unwrap(), File::open(V2).unwrap());
    for name in ["values", "grp/inner"] {
        assert_eq!(
            a.dataset(name).unwrap().read_f64().unwrap(),
            b.dataset(name).unwrap().read_f64().unwrap(),
            "{name} differs between the two superblock versions"
        );
    }
    assert_eq!(
        a.dataset("chunked").unwrap().read_i32().unwrap(),
        b.dataset("chunked").unwrap().read_i32().unwrap()
    );
    assert_eq!(a.root().attrs().unwrap(), b.root().attrs().unwrap());
}

/// The superblock version byte, located by scanning for the signature so a
/// userblock would not throw the offset off.
fn superblock_version(path: &str) -> u8 {
    let bytes = std::fs::read(path).unwrap();
    let sig = bytes
        .windows(8)
        .position(|w| w == b"\x89HDF\r\n\x1a\n")
        .expect("the fixture carries an HDF5 signature");
    bytes[sig + 8]
}
