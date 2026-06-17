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
    // hdf5-pure repacks it (a chunked dataset, so its compressed chunks are copied
    // verbatim with the source filter-pipeline message carried through), and the
    // reference C library decodes the result. That the C library reads the exact
    // values back proves the verbatim-copied chunk format and reused pipeline
    // message are valid and interoperable, not a layout that merely round-trips
    // in-crate.
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

#[test]
fn c_deflate_dataset_repacked_verbatim_read_by_both() {
    // The reference C library writes a chunked + shuffle + deflate dataset.
    // hdf5-pure repacks it by copying the C library's *own* compressed chunk
    // streams verbatim (it never re-encodes them), then both readers decode the
    // result. That the C library reads its exact values back proves the copied
    // chunk bytes and the carried-through filter-pipeline message remain valid.
    let dir = tempdir().unwrap();
    let src = dir.path().join("c_deflate.h5");
    let dst = dir.path().join("c_deflate_repacked.h5");

    let data: Vec<i32> = (0..4096).map(|i| i % 13).collect();
    {
        let file = hdf5::File::create(&src).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape((4096,))
            .chunk((512,))
            .shuffle()
            .deflate(6)
            .create("vals")
            .unwrap();
        ds.write_raw(data.as_slice()).unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = File::open(&dst).unwrap();
    assert_eq!(f.dataset("vals").unwrap().read_i32().unwrap(), data);

    let c = hdf5::File::open(&dst).unwrap();
    assert_eq!(c.dataset("vals").unwrap().read_raw::<i32>().unwrap(), data);
}

#[test]
fn c_sparse_chunked_lossless_repacked_falls_back() {
    // The C library writes only the first chunks of a chunked dataset, leaving the
    // tail unallocated (a sparse chunk grid). The verbatim path needs a dense
    // grid, so repack falls back to the read-raw + re-encode path. With a lossless
    // pipeline (deflate) that fallback is faithful: the written values survive and
    // the unwritten tail reads back as the fill value (0).
    let dir = tempdir().unwrap();
    let src = dir.path().join("c_sparse.h5");
    let dst = dir.path().join("c_sparse_repacked.h5");

    let n = 2000usize; // chunk 512 -> 4 chunks; only the first ~1000 written
    let written = 1000usize;
    let head: Vec<i32> = (1..=written as i32).collect();
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .create(&src)
            .unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([n])
            .chunk([512])
            .deflate(4)
            .create("data")
            .unwrap();
        ds.write_slice(head.as_slice(), 0..written).unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = File::open(&dst).unwrap();
    let vals = f.dataset("data").unwrap().read_i32().unwrap();
    assert_eq!(vals.len(), n);
    assert_eq!(&vals[..written], head.as_slice());
    assert!(
        vals[written..].iter().all(|&v| v == 0),
        "unwritten tail should read back as the fill value"
    );
}

#[test]
fn c_sparse_chunked_lossy_repack_refused() {
    // The C library writes only the first chunks of a chunked dataset compressed
    // with float D-scale scale-offset (a lossy filter), leaving the tail
    // unallocated (a sparse grid). The verbatim path needs a dense grid, so repack
    // would fall back to the read-raw + re-encode path — but re-encoding a lossy
    // filter is not guaranteed idempotent, so repack must refuse by name rather
    // than risk silently perturbing the data.
    use hdf5::filters::ScaleOffset as CScaleOffset;

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_sparse_lossy.h5");
    let dst = dir.path().join("c_sparse_lossy_repacked.h5");

    let n = 2000usize;
    let written = 1000usize;
    let head: Vec<f64> = (0..written).map(|i| i as f64 * 0.01).collect();
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .create(&src)
            .unwrap();
        let ds = file
            .new_dataset::<f64>()
            .shape([n])
            .chunk([512])
            .scale_offset(CScaleOffset::FloatDScale(3))
            .create("data")
            .unwrap();
        ds.write_slice(head.as_slice(), 0..written).unwrap();
        file.close().unwrap();
    }

    let err = repack(&src, &dst, &RepackOptions::new()).unwrap_err();
    match err {
        hdf5_pure::Error::RepackUnsupported(msg) => assert!(
            msg.contains("data") && msg.contains("scale-offset"),
            "error should name the dataset and reason: {msg}"
        ),
        other => panic!("expected RepackUnsupported, got {other:?}"),
    }
    assert!(!dst.exists(), "dst must not be created when repack refuses");
}
