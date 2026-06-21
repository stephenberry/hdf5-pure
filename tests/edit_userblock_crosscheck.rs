// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit-pointer targets; skip them on 32-bit so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(not(target_pointer_width = "32"))]
//! Cross-validation for editing files with a userblock (the userblock slice of
//! issue #104) against the reference C library. A single off-by-base address in
//! an edited file makes the C library error or read garbage, so reading the
//! result back through `hdf5` is the external tripwire that the editor stored
//! every root/link/layout address relative to the base address correctly.

use hdf5_pure::{EditSession, File, FileBuilder};
use tempfile::tempdir;

const UB: u64 = 512;

#[test]
fn pure_userblock_file_edited_then_read_by_c_library() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("pure_ub.h5");

    let mut b = FileBuilder::new();
    b.with_userblock(UB);
    b.create_dataset("alpha").with_f64_data(&[1.0, 2.0, 3.0]);
    let mut g = b.create_group("grp");
    g.create_dataset("beta").with_i32_data(&[10, 20, 30, 40]);
    b.add_group(g.finish());
    std::fs::write(&path, b.finish().unwrap()).unwrap();

    {
        let mut s = EditSession::open(&path).unwrap();
        s.write_dataset("alpha").with_f64_data(&[9.0, 8.0, 7.0]);
        s.create_dataset("added").with_f64_data(&[100.0, 200.0]);
        s.create_dataset("grp/gamma").with_i32_data(&[1, 2, 3]);
        s.commit().unwrap();
    }

    // pure reader
    let f = File::open(&path).unwrap();
    assert_eq!(f.dataset("alpha").unwrap().read_f64().unwrap(), vec![9.0, 8.0, 7.0]);
    assert_eq!(f.dataset("added").unwrap().read_f64().unwrap(), vec![100.0, 200.0]);
    assert_eq!(f.dataset("grp/gamma").unwrap().read_i32().unwrap(), vec![1, 2, 3]);

    // reference C library reader — the real interop proof.
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(c.dataset("alpha").unwrap().read_raw::<f64>().unwrap(), vec![9.0, 8.0, 7.0]);
    assert_eq!(c.dataset("added").unwrap().read_raw::<f64>().unwrap(), vec![100.0, 200.0]);
    assert_eq!(
        c.dataset("grp/beta").unwrap().read_raw::<i32>().unwrap(),
        vec![10, 20, 30, 40]
    );
    assert_eq!(c.dataset("grp/gamma").unwrap().read_raw::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn real_mat_edited_then_read_by_c_library() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("real.mat");
    std::fs::copy("tests/fixtures/mat_real/test_string_v73.mat", &path).unwrap();

    let before = File::open(&path)
        .unwrap()
        .dataset("string_scalar")
        .unwrap()
        .read_u32()
        .unwrap();

    {
        let mut s = EditSession::open(&path).unwrap();
        s.create_dataset("hdf5_pure_probe").with_f64_data(&[42.0, -1.0, 3.5]);
        s.commit().unwrap();
    }

    // The C library opens the edited real .mat file and reads both the added
    // dataset and an untouched original — proving every address survived the edit
    // relative to the 512-byte MATLAB userblock.
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("hdf5_pure_probe").unwrap().read_raw::<f64>().unwrap(),
        vec![42.0, -1.0, 3.5]
    );
    assert_eq!(
        c.dataset("string_scalar").unwrap().read_raw::<u32>().unwrap(),
        before
    );
}
