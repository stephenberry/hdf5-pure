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
fn c_reads_repacked_scale_offset() {
    // hdf5-pure writes an integer dataset compressed with lossless scale-offset,
    // hdf5-pure repacks it, and the reference C library decodes the *re-emitted*
    // filter. That the C library reads the exact values back proves repack's
    // re-applied scale-offset chunk format is valid and interoperable, not a
    // malformed pipeline that merely happens to round-trip in-crate.
    let dir = tempdir().unwrap();
    let src = dir.path().join("so_src.h5");
    let dst = dir.path().join("so_repacked.h5");

    let data: Vec<i32> = (0..1024).map(|i| 100 + i % 17).collect();
    {
        let mut b = hdf5_pure::FileBuilder::new();
        b.create_dataset("vals")
            .with_i32_data(&data)
            .with_chunks(&[128])
            .with_scale_offset(hdf5_pure::ScaleOffset::Integer(0));
        b.write(&src).unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = File::open(&dst).unwrap();
    assert_eq!(f.dataset("vals").unwrap().read_i32().unwrap(), data);

    let c = hdf5::File::open(&dst).unwrap();
    assert_eq!(c.dataset("vals").unwrap().read_raw::<i32>().unwrap(), data);
}

#[test]
fn repack_roundtrips_c_vlen_string_dataset() {
    use hdf5::types::VarLenUnicode;
    use std::str::FromStr;

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_vlen.h5");
    let dst = dir.path().join("vlen_repacked.h5");

    let words = ["alpha", "beta", "gamma", "", "δelta"];
    {
        let file = hdf5::File::create(&src).unwrap();
        let vals: Vec<VarLenUnicode> = words
            .iter()
            .map(|s| VarLenUnicode::from_str(s).unwrap())
            .collect();
        file.new_dataset::<VarLenUnicode>()
            .shape((words.len(),))
            .create("labels")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    // A VL-string dataset the C library wrote must now round-trip faithfully.
    repack(&src, &dst, &RepackOptions::new()).unwrap();

    // hdf5-pure reads the repacked values back, in order, including the empty
    // and non-ASCII elements.
    let f = File::open(&dst).unwrap();
    let labels = f.dataset("labels").unwrap();
    let got = labels
        .read_vlen_strings(Default::default())
        .unwrap();
    assert_eq!(got, words);

    // The datatype must remain variable-length, not be silently converted to a
    // fixed-length string.
    assert!(
        matches!(
            labels.datatype().unwrap(),
            hdf5_pure::Datatype::VariableLength { .. }
        ),
        "repacked datatype must stay variable-length"
    );

    // The reference C library agrees on both values and that the datatype is
    // variable-length — the real interop proof.
    let c = hdf5::File::open(&dst).unwrap();
    let cds = c.dataset("labels").unwrap();
    let cvals = cds.read_raw::<VarLenUnicode>().unwrap();
    let cstrings: Vec<String> = cvals.iter().map(|v| v.as_str().to_string()).collect();
    assert_eq!(cstrings, words);
    assert!(
        cds.dtype().unwrap().is::<VarLenUnicode>(),
        "C library must see a variable-length Unicode string datatype"
    );
}

#[test]
fn repack_roundtrips_vlen_string_2d() {
    use hdf5::types::VarLenUnicode;
    use std::str::FromStr;

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_vlen_2d.h5");
    let dst = dir.path().join("vlen_2d_repacked.h5");

    // 2x3 grid, row-major.
    let words = ["a", "bb", "ccc", "", "ee", "ffffff"];
    {
        let file = hdf5::File::create(&src).unwrap();
        let vals: Vec<VarLenUnicode> = words
            .iter()
            .map(|s| VarLenUnicode::from_str(s).unwrap())
            .collect();
        file.new_dataset::<VarLenUnicode>()
            .shape((2, 3))
            .create("grid")
            .unwrap()
            .write_raw(&vals)
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = File::open(&dst).unwrap();
    let grid = f.dataset("grid").unwrap();
    assert_eq!(grid.shape().unwrap(), vec![2, 3]);
    assert_eq!(
        grid.read_vlen_strings(Default::default()).unwrap(),
        words
    );

    // C library agrees on shape and values.
    let c = hdf5::File::open(&dst).unwrap();
    let cds = c.dataset("grid").unwrap();
    assert_eq!(cds.shape(), vec![2, 3]);
    let cvals = cds.read_raw::<VarLenUnicode>().unwrap();
    let cstrings: Vec<String> = cvals.iter().map(|v| v.as_str().to_string()).collect();
    assert_eq!(cstrings, words);
}

#[test]
fn repack_refuses_chunked_vlen_string_dataset() {
    use hdf5::types::VarLenUnicode;
    use std::str::FromStr;

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_vlen_chunked.h5");
    let dst = dir.path().join("vlen_chunked_repacked.h5");

    {
        let file = hdf5::File::create(&src).unwrap();
        let vals: Vec<VarLenUnicode> = (0..8)
            .map(|i| VarLenUnicode::from_str(&format!("word{i}")).unwrap())
            .collect();
        file.new_dataset::<VarLenUnicode>()
            .shape((8,))
            .chunk((4,))
            .create("chunked_labels")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    // Chunked VL-string datasets cannot be repacked faithfully (their element
    // references live inside compressed chunks before the heap addresses are
    // known), so repack must refuse by name.
    let err = repack(&src, &dst, &RepackOptions::new()).unwrap_err();
    match err {
        hdf5_pure::Error::RepackUnsupported(msg) => {
            assert!(
                msg.contains("chunked_labels") && msg.contains("chunked"),
                "error should name the dataset and reason: {msg}"
            );
        }
        other => panic!("expected RepackUnsupported, got {other:?}"),
    }
    assert!(!dst.exists(), "dst must not be created when repack refuses");
}

#[test]
fn repack_refuses_unrepresentable_attribute() {
    // The C library writes a boolean attribute, which is an HDF5 enumeration —
    // a datatype the reader cannot decode into an AttrValue and would silently
    // drop. Repack must refuse by name rather than write a file missing the
    // attribute, upholding the fail-loud fidelity contract.
    let dir = tempdir().unwrap();
    let src = dir.path().join("c_boolattr.h5");
    let dst = dir.path().join("boolattr_repacked.h5");

    {
        let file = hdf5::File::create(&src).unwrap();
        let ds = file
            .new_dataset::<f64>()
            .shape((2,))
            .create("data")
            .unwrap();
        ds.write(&[1.0f64, 2.0]).unwrap();
        // A boolean (enum) attribute the pure reader cannot represent.
        ds.new_attr::<bool>()
            .shape(())
            .create("active")
            .unwrap()
            .write_scalar(&true)
            .unwrap();
        file.close().unwrap();
    }

    let err = repack(&src, &dst, &RepackOptions::new()).unwrap_err();
    match err {
        hdf5_pure::Error::RepackUnsupported(msg) => {
            assert!(
                msg.contains("active") && msg.contains("data"),
                "error should name the attribute and its dataset: {msg}"
            );
        }
        other => panic!("expected RepackUnsupported, got {other:?}"),
    }
    assert!(!dst.exists(), "dst must not be created when repack refuses");
}
