//! Write one file in each on-disk format this crate produces, for checking
//! against an HDF5 1.8 library.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example libver_fixtures --features serde -- <out-dir>
//! ```
//!
//! `scripts/check-hdf5-18.sh` drives this; it is a separate program rather than
//! a test because the thing it feeds is an external toolchain that cannot be a
//! dev-dependency (see that script for why).
//!
//! Both files hold the same content, so any difference an old library reports
//! between them is the format and nothing else.

use hdf5_pure::mat::{self, Options};
use hdf5_pure::{AttrValue, FileBuilder, LibVer, make_i32_type};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Serialize)]
struct Demo {
    values: Vec<f64>,
    label: String,
    nested: Inner,
    empty: Vec<f64>,
    /// Ragged, so it lowers to a cell array rather than a matrix. This is the
    /// only shape that interns objects under `#refs#`, where each one carries
    /// an `H5PATH` attribute and the parent dataset holds object references
    /// rather than data — two things no other fixture here puts in front of an
    /// old library.
    ragged: Vec<Vec<i32>>,
}

#[derive(Serialize)]
struct Inner {
    count: u32,
    flag: bool,
}

fn demo() -> Demo {
    Demo {
        values: vec![1.0, 2.0, 3.0],
        label: "demo".to_string(),
        nested: Inner {
            count: 7,
            flag: true,
        },
        empty: Vec::new(),
        ragged: vec![vec![1], vec![2, 3]],
    }
}

/// A plain `.h5` alongside the `.mat` pair: attributes on all three kinds of
/// object, which is what the attribute-count fix touches, and a committed
/// datatype, which is an object kind of its own.
fn write_h5(path: &Path, libver: LibVer) {
    let mut b = FileBuilder::new();
    b.with_libver_bounds(LibVer::Earliest, libver);
    b.set_attr("root_attr", AttrValue::AsciiString("r".into()));
    b.create_dataset("values")
        .with_f64_data(&[1.0, 2.0, 3.0])
        .set_attr("units", AttrValue::AsciiString("m/s".into()));

    // A committed (`H5Tcommit`) datatype, named by a dataset and by an
    // attribute. Its users carry a reference to its object header in place of an
    // encoding, and the object itself carries a reference count — a shape no
    // other fixture here writes, and one an old library has to decode rather
    // than merely skip. A reference it cannot follow costs more than the type's
    // name: the C library abandons an object's whole attribute list when one
    // attribute fails to decode, so the repack count below sees it too.
    b.commit_datatype("reading_t", make_i32_type());
    b.create_dataset("typed")
        .with_i32_data(&[3, 1, 4])
        .with_committed_datatype("reading_t")
        .set_attr_committed("baseline", AttrValue::I32(9), "reading_t");

    let mut g = b.create_group("grp");
    g.set_attr("tag", AttrValue::I64(7));
    g.create_dataset("inner").with_i32_data(&[7, 8]);
    let g = g.finish();
    b.add_group(g);
    b.write(path).expect("write h5 fixture");
}

fn superblock_version(path: &Path) -> u8 {
    let bytes = std::fs::read(path).expect("read fixture");
    let sig = bytes
        .windows(8)
        .position(|w| w == b"\x89HDF\r\n\x1a\n")
        .expect("fixture carries an HDF5 signature");
    bytes[sig + 8]
}

fn main() {
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "libver_fixtures".to_string()),
    );
    std::fs::create_dir_all(&out).expect("create output dir");

    // The MAT default since 0.34: the HDF5 1.8 format.
    let mat_v18 = out.join("mat_v18.mat");
    mat::to_file(&demo(), &mat_v18).expect("write 1.8 mat");

    // What every release through 0.33.0 wrote.
    let mat_v110 = out.join("mat_v110.mat");
    let mut opts = Options::default();
    opts.libver = LibVer::V110;
    mat::to_file_with_options(&demo(), &mat_v110, &opts).expect("write 1.10 mat");

    let h5_v18 = out.join("plain_v18.h5");
    write_h5(&h5_v18, LibVer::V18);
    let h5_v110 = out.join("plain_v110.h5");
    write_h5(&h5_v110, LibVer::V110);

    for path in [&mat_v18, &mat_v110, &h5_v18, &h5_v110] {
        println!("{} superblock {}", path.display(), superblock_version(path));
    }
}
