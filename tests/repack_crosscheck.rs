// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit-pointer targets; skip them on 32-bit so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(not(target_pointer_width = "32"))]
//! Cross-validation for whole-file repack (issue #21) against the reference
//! HDF5 C library: a file the C library *writes* is repacked by `hdf5_pure`,
//! and the result is read back by both readers. Also proves the fail-loud
//! contract on a real variable-length string dataset the C library produces.

use hdf5_pure::{File, RepackOptions, repack};
use tempfile::tempdir;

#[test]
fn c_file_repacked_then_read_by_c_library() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("c_src.h5");
    let dst = dir.path().join("repacked.h5");

    // The C library writes alpha (f64), doomed (i32, to be dropped), and a group
    // grp/beta (i32), using the 1.10+ format.
    {
        let file = hdf5::File::create(&src).unwrap();
        file.new_dataset::<f64>()
            .shape((3,))
            .create("alpha")
            .unwrap()
            .write(&[1.0f64, 2.0, 3.0])
            .unwrap();
        file.new_dataset::<i32>()
            .shape((4,))
            .create("doomed")
            .unwrap()
            .write(&[7i32, 8, 9, 10])
            .unwrap();
        let grp = file.create_group("grp").unwrap();
        grp.new_dataset::<i32>()
            .shape((4,))
            .create("beta")
            .unwrap()
            .write(&[10i32, 20, 30, 40])
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new().drop_path("doomed")).unwrap();

    // hdf5-pure reads the repacked file: survivors intact, dropped gone.
    let f = File::open(&dst).unwrap();
    assert_eq!(
        f.dataset("alpha").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        f.dataset("grp/beta").unwrap().read_i32().unwrap(),
        vec![10, 20, 30, 40]
    );
    assert!(f.dataset("doomed").is_err());

    // The reference C library agrees — the real interop proof.
    let c = hdf5::File::open(&dst).unwrap();
    assert_eq!(
        c.dataset("alpha").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        c.dataset("grp/beta").unwrap().read_raw::<i32>().unwrap(),
        vec![10, 20, 30, 40]
    );
    assert!(
        c.dataset("doomed").is_err(),
        "dropped dataset still present (C library)"
    );
}

#[test]
fn repack_refuses_c_vlen_string_dataset() {
    use hdf5::types::VarLenUnicode;
    use std::str::FromStr;

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_vlen.h5");
    let dst = dir.path().join("vlen_repacked.h5");

    {
        let file = hdf5::File::create(&src).unwrap();
        let words: Vec<VarLenUnicode> = ["alpha", "beta", "gamma"]
            .iter()
            .map(|s| VarLenUnicode::from_str(s).unwrap())
            .collect();
        file.new_dataset::<VarLenUnicode>()
            .shape((3,))
            .create("labels")
            .unwrap()
            .write(&words)
            .unwrap();
        file.close().unwrap();
    }

    // A variable-length string dataset cannot be re-emitted faithfully yet, so
    // repack must refuse by name rather than silently degrade it.
    let err = repack(&src, &dst, &RepackOptions::new()).unwrap_err();
    match err {
        hdf5_pure::Error::RepackUnsupported(msg) => {
            assert!(
                msg.contains("labels") && msg.contains("variable-length"),
                "error should name the dataset and reason: {msg}"
            );
        }
        other => panic!("expected RepackUnsupported, got {other:?}"),
    }
    assert!(!dst.exists(), "dst must not be created when repack refuses");
}
