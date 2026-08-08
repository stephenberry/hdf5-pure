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

mod common;
use common::assert_c_absent;

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
    assert_c_absent(&c.dataset("doomed").unwrap_err(), "doomed");
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
    let got = labels.read_vlen_strings(Default::default()).unwrap();
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

/// The C library fills a collection and allocates another, so a large enough
/// variable-length dataset it writes already spans several of them. Repack
/// re-stages that data through fresh collections of its own, which is only
/// faithful if both the read and the write sides carry each element's
/// collection along with its index.
#[test]
fn repack_roundtrips_c_vlen_dataset_spanning_many_collections() {
    use hdf5::types::VarLenUnicode;
    use std::str::FromStr;

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_vlen_many.h5");
    let dst = dir.path().join("vlen_many_repacked.h5");

    // Past one collection's 65,535-object index, so the destination splits too.
    let count = u16::MAX as usize + 5_000;
    let words: Vec<String> = (0..count).map(|i| format!("s{i}")).collect();
    {
        let file = hdf5::File::create(&src).unwrap();
        let vals: Vec<VarLenUnicode> = words
            .iter()
            .map(|s| VarLenUnicode::from_str(s).unwrap())
            .collect();
        file.new_dataset::<VarLenUnicode>()
            .shape((count,))
            .create("labels")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = File::open(&dst).unwrap();
    let got = f
        .dataset("labels")
        .unwrap()
        .read_vlen_strings(Default::default())
        .unwrap();
    assert_eq!(got, words);

    // The reference library agrees, element for element.
    let c = hdf5::File::open(&dst).unwrap();
    let cvals = c
        .dataset("labels")
        .unwrap()
        .read_raw::<VarLenUnicode>()
        .unwrap();
    assert_eq!(cvals.len(), count);
    for i in [0, 65_534, 65_535, count - 1] {
        assert_eq!(cvals[i].as_str(), words[i], "element {i} differs");
    }
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
    assert_eq!(grid.read_vlen_strings(Default::default()).unwrap(), words);

    // C library agrees on shape and values.
    let c = hdf5::File::open(&dst).unwrap();
    let cds = c.dataset("grid").unwrap();
    assert_eq!(cds.shape(), vec![2, 3]);
    let cvals = cds.read_raw::<VarLenUnicode>().unwrap();
    let cstrings: Vec<String> = cvals.iter().map(|v| v.as_str().to_string()).collect();
    assert_eq!(cstrings, words);
}

#[test]
fn repack_roundtrips_c_vlen_sequence_dataset() {
    use hdf5::types::VarLenArray;

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_vlen_seq.h5");
    let dst = dir.path().join("vlen_seq_repacked.h5");

    // The C library writes a non-string VL dataset (`H5T_VLEN { i32 }`),
    // including an empty sequence.
    let seqs: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![], vec![-7, 42, 0, 99], vec![5]];
    {
        let file = hdf5::File::create(&src).unwrap();
        let vals: Vec<VarLenArray<i32>> = seqs
            .iter()
            .map(|s| VarLenArray::from_slice(s.as_slice()))
            .collect();
        file.new_dataset::<VarLenArray<i32>>()
            .shape((seqs.len(),))
            .create("seqs")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    // hdf5-pure must see the repacked datatype as a non-string variable-length
    // sequence (not silently converted).
    let f = File::open(&dst).unwrap();
    let ds = f.dataset("seqs").unwrap();
    assert!(
        matches!(
            ds.datatype().unwrap(),
            hdf5_pure::Datatype::VariableLength {
                is_string: false,
                ..
            }
        ),
        "repacked datatype must stay a non-string variable-length sequence"
    );

    // The reference C library reads the exact sequences back — the interop proof
    // that the re-staged global heap and rebuilt references are valid.
    let c = hdf5::File::open(&dst).unwrap();
    let cds = c.dataset("seqs").unwrap();
    let cvals = cds.read_raw::<VarLenArray<i32>>().unwrap();
    let got: Vec<Vec<i32>> = cvals.iter().map(|v| v.as_slice().to_vec()).collect();
    assert_eq!(got, seqs);
}

#[test]
fn repack_roundtrips_c_object_reference_dataset() {
    use hdf5::{ObjectReference, ObjectReference1, ReferencedObject};

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_refs.h5");
    let dst = dir.path().join("refs_repacked.h5");

    // The C library writes two targets plus a dataset of object references to
    // them (one in the root, one in a subgroup).
    {
        let file = hdf5::File::create(&src).unwrap();
        file.new_dataset::<f64>()
            .shape((3,))
            .create("alpha")
            .unwrap()
            .write(&[1.0f64, 2.0, 3.0])
            .unwrap();
        let grp = file.create_group("grp").unwrap();
        grp.new_dataset::<i32>()
            .shape((4,))
            .create("beta")
            .unwrap()
            .write(&[10i32, 20, 30, 40])
            .unwrap();
        let refs = vec![
            ObjectReference1::create(&file, "alpha").unwrap(),
            ObjectReference1::create(&file, "grp/beta").unwrap(),
            // A reference to the root group exercises the empty-path resolution.
            ObjectReference1::create(&file, "/").unwrap(),
        ];
        file.new_dataset::<ObjectReference1>()
            .shape((3,))
            .create("refs")
            .unwrap()
            .write(&refs)
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    // hdf5-pure sees the repacked datatype as an object reference (not converted).
    let f = File::open(&dst).unwrap();
    assert!(
        matches!(
            f.dataset("refs").unwrap().datatype().unwrap(),
            hdf5_pure::Datatype::Reference {
                ref_type: hdf5_pure::ReferenceType::Object,
                ..
            }
        ),
        "repacked datatype must stay an object reference"
    );

    // The reference C library dereferences each repacked reference to the right
    // object — the proof that the addresses were rewritten to the new locations
    // rather than copied stale.
    let c = hdf5::File::open(&dst).unwrap();
    let cds = c.dataset("refs").unwrap();
    let cvals = cds.read_raw::<ObjectReference1>().unwrap();
    assert_eq!(cvals.len(), 3);
    match cvals[0].dereference(&c).unwrap() {
        ReferencedObject::Dataset(d) => {
            assert_eq!(d.read_raw::<f64>().unwrap(), vec![1.0, 2.0, 3.0]);
        }
        other => panic!("ref 0 should resolve to a dataset, got {other:?}"),
    }
    match cvals[1].dereference(&c).unwrap() {
        ReferencedObject::Dataset(d) => {
            assert_eq!(d.read_raw::<i32>().unwrap(), vec![10, 20, 30, 40]);
        }
        other => panic!("ref 1 should resolve to a dataset, got {other:?}"),
    }
    match cvals[2].dereference(&c).unwrap() {
        ReferencedObject::Group(_) => {}
        other => panic!("ref 2 should resolve to the root group, got {other:?}"),
    }
}

#[test]
fn repack_refuses_reference_to_dropped_object() {
    use hdf5::{ObjectReference, ObjectReference1};

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_refs_drop.h5");
    let dst = dir.path().join("refs_drop_repacked.h5");

    {
        let file = hdf5::File::create(&src).unwrap();
        file.new_dataset::<f64>()
            .shape((2,))
            .create("target")
            .unwrap()
            .write(&[1.0f64, 2.0])
            .unwrap();
        let refs = vec![ObjectReference1::create(&file, "target").unwrap()];
        file.new_dataset::<ObjectReference1>()
            .shape((1,))
            .create("refs")
            .unwrap()
            .write(&refs)
            .unwrap();
        file.close().unwrap();
    }

    // Dropping the referenced object must fail the repack by name rather than
    // silently leave a dangling reference.
    let err = repack(&src, &dst, &RepackOptions::new().drop_path("target")).unwrap_err();
    match err {
        hdf5_pure::Error::RepackUnsupported(msg) => assert!(
            msg.contains("refs") && msg.contains("target") && msg.contains("dropped"),
            "error should name the reference dataset and the dropped target: {msg}"
        ),
        other => panic!("expected RepackUnsupported, got {other:?}"),
    }
    assert!(!dst.exists(), "dst must not be created when repack refuses");
}

/// A chunked VL-string dataset the C library wrote repacks into an equally
/// chunked one, with its heap references rebuilt against the new file's
/// collections (issue #109). The C library reading the *output* is the check
/// that those rebuilt addresses are real.
#[test]
fn repack_roundtrips_chunked_vlen_string_dataset() {
    use hdf5::types::VarLenUnicode;
    use std::str::FromStr;

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_vlen_chunked.h5");
    let dst = dir.path().join("vlen_chunked_repacked.h5");

    let expected: Vec<String> = (0..8).map(|i| format!("word{i}")).collect();
    {
        let file = hdf5::File::create(&src).unwrap();
        let vals: Vec<VarLenUnicode> = expected
            .iter()
            .map(|s| VarLenUnicode::from_str(s).unwrap())
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

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    assert_eq!(
        File::open(&dst)
            .unwrap()
            .dataset("chunked_labels")
            .unwrap()
            .read_string()
            .unwrap(),
        expected
    );
    let f = hdf5::File::open(&dst).unwrap();
    let ds = f.dataset("chunked_labels").unwrap();
    assert!(
        ds.is_chunked(),
        "repack must reproduce the source's chunked layout, not flatten it"
    );
    let read: Vec<String> = ds
        .read_raw::<VarLenUnicode>()
        .unwrap()
        .into_iter()
        .map(|s| s.as_str().to_owned())
        .collect();
    assert_eq!(read, expected);
}

/// The compressed and resizable variants of the same path, written by this crate
/// and repacked: the pipeline and the unlimited dimension must survive, and the
/// C library must read the result.
#[test]
fn repack_roundtrips_filtered_and_resizable_vlen_string_datasets() {
    use hdf5::types::VarLenUnicode;

    let dir = tempdir().unwrap();
    let src = dir.path().join("filtered_vlen.h5");
    let dst = dir.path().join("filtered_vlen_repacked.h5");

    let words: Vec<String> = (0..40).map(|i| format!("label-{i}")).collect();
    let refs: Vec<&str> = words.iter().map(String::as_str).collect();
    {
        let mut b = hdf5_pure::FileBuilder::new();
        b.create_dataset("compressed")
            .with_vlen_strings(&refs)
            .with_chunks(&[8])
            .with_deflate(6);
        b.create_dataset("growable")
            .with_vlen_strings(&refs)
            .with_shape(&[words.len() as u64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[8]);
        b.write(&src).unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let out = File::open(&dst).unwrap();
    for name in ["compressed", "growable"] {
        assert_eq!(
            out.dataset(name).unwrap().read_string().unwrap(),
            words,
            "{name} did not survive the repack"
        );
    }

    let f = hdf5::File::open(&dst).unwrap();
    let compressed = f.dataset("compressed").unwrap();
    assert!(
        !compressed.filters().is_empty(),
        "repack must carry the deflate pipeline onto the rebuilt dataset"
    );
    assert!(
        f.dataset("growable").unwrap().is_resizable(),
        "repack must carry the unlimited dimension onto the rebuilt dataset"
    );
    for name in ["compressed", "growable"] {
        let read: Vec<String> = f
            .dataset(name)
            .unwrap()
            .read_raw::<VarLenUnicode>()
            .unwrap()
            .into_iter()
            .map(|s| s.as_str().to_owned())
            .collect();
        assert_eq!(read, words, "C library disagreed on {name}");
    }
}

/// A boolean attribute is an HDF5 enumeration, which [`AttrValue`] has no
/// variant for. Repack used to refuse the whole file over one — the only honest
/// answer while every attribute went through a decode, since the alternative was
/// dropping it silently.
///
/// Copying the message verbatim answers it properly instead: an enumeration
/// carries no address, so its bytes mean the same thing in the destination and
/// the attribute simply survives. This is the general form of the refusal that
/// went away, not a special case for booleans — a compound or opaque attribute
/// travels for the same reason (issue #241).
#[test]
fn repack_carries_an_attribute_no_attr_value_can_express() {
    use hdf5::types::TypeDescriptor;

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
        ds.new_attr::<bool>()
            .shape(())
            .create("active")
            .unwrap()
            .write_scalar(&true)
            .unwrap();
        file.close().unwrap();
    }

    let source_type = {
        let c = hdf5::File::open(&src).unwrap();
        c.dataset("data")
            .unwrap()
            .attr("active")
            .unwrap()
            .dtype()
            .unwrap()
            .to_descriptor()
            .unwrap()
    };
    assert!(
        matches!(source_type, TypeDescriptor::Boolean),
        "the C library should have written an enumeration: {source_type:?}"
    );

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let c = hdf5::File::open(&dst).unwrap();
    let attr = c.dataset("data").unwrap().attr("active").unwrap();
    assert_eq!(
        attr.dtype().unwrap().to_descriptor().unwrap(),
        source_type,
        "the enumeration must cross the repack as itself"
    );
    assert!(
        attr.read_scalar::<bool>().unwrap(),
        "and still hold its value"
    );
}

/// What the fail-loud contract still covers once verbatim copying has taken the
/// address-free datatypes off the refusal list.
///
/// An object-reference attribute stores an object-header address, so its bytes
/// cannot be copied, and [`AttrValue`] has no variant to re-encode it from
/// either. Repack must still name it and refuse rather than write a file the
/// attribute is missing from — dropping it silently is the failure this test
/// exists to catch, and the shrinking refusal list makes it the last one holding
/// that line for attributes.
#[test]
fn repack_refuses_an_attribute_whose_address_it_cannot_rewrite() {
    use hdf5::{ObjectReference, ObjectReference1};

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_refattr.h5");
    let dst = dir.path().join("refattr_repacked.h5");

    {
        let file = hdf5::File::create(&src).unwrap();
        let ds = file
            .new_dataset::<f64>()
            .shape((2,))
            .create("data")
            .unwrap();
        ds.write(&[1.0f64, 2.0]).unwrap();
        ds.new_attr::<ObjectReference1>()
            .shape(())
            .create("points_at")
            .unwrap()
            .write_scalar(&ObjectReference1::create(&file, "data").unwrap())
            .unwrap();
        file.close().unwrap();
    }

    let err = repack(&src, &dst, &RepackOptions::new()).unwrap_err();
    match err {
        hdf5_pure::Error::RepackUnsupported(msg) => {
            assert!(
                msg.contains("points_at") && msg.contains("data"),
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

/// A compound whose members include a variable-length string stores a 16-byte
/// global-heap reference *inside* each element. Copying those element bytes
/// verbatim leaves the reference pointing at a collection in the source file's
/// address space, which the destination never allocates (issue #201). Repack has
/// to re-stage the payloads through the destination's own heap and rewrite the
/// embedded references, exactly as it does for a top-level VL dataset.
///
/// The C library is the load-bearing reader here: this crate's own reader
/// resolves the heap lazily and reads `read_raw` bytes without validating the
/// embedded addresses, so a self-round-trip alone would not catch a stale one.
#[test]
fn repack_roundtrips_c_compound_with_vlen_string_member() {
    use hdf5::types::VarLenUnicode;
    use std::str::FromStr;

    #[derive(hdf5::H5Type, Clone, Debug, PartialEq)]
    #[repr(C)]
    struct Labelled {
        id: i32,
        label: VarLenUnicode,
    }

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_compound_vlen.h5");
    let dst = dir.path().join("compound_vlen_repacked.h5");

    let rows = [
        "alpha",
        "",
        "gamma",
        "δelta",
        "a longer label than the rest",
    ];
    {
        let file = hdf5::File::create(&src).unwrap();
        let vals: Vec<Labelled> = rows
            .iter()
            .enumerate()
            .map(|(i, s)| Labelled {
                id: i as i32 * 10,
                label: VarLenUnicode::from_str(s).unwrap(),
            })
            .collect();
        file.new_dataset::<Labelled>()
            .shape((rows.len(),))
            .create("rows")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    // The datatype must stay a compound carrying a variable-length member, not
    // be flattened or converted.
    let f = File::open(&dst).unwrap();
    let ds = f.dataset("rows").unwrap();
    match ds.datatype().unwrap() {
        hdf5_pure::Datatype::Compound { members, .. } => {
            assert_eq!(members.len(), 2);
            assert!(
                matches!(
                    members[1].datatype,
                    hdf5_pure::Datatype::VariableLength { .. }
                ),
                "the label member must stay variable-length"
            );
        }
        other => panic!("expected a compound datatype, got {other:?}"),
    }

    // The reference C library reads every row back — the interop proof that the
    // embedded heap references name collections in the *destination*.
    let c = hdf5::File::open(&dst).unwrap();
    let cvals = c.dataset("rows").unwrap().read_raw::<Labelled>().unwrap();
    assert_eq!(cvals.len(), rows.len());
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(cvals[i].id, i as i32 * 10, "row {i} id differs");
        assert_eq!(cvals[i].label.as_str(), *row, "row {i} label differs");
    }
}

/// A compound can carry several variable-length members of different kinds, and
/// each one's reference stores a *count* in the units of its own base type — bytes
/// for a string, elements for a sequence. Re-staging has to keep those units
/// straight per member, not per dataset.
#[test]
fn repack_roundtrips_c_compound_with_two_vlen_members() {
    use hdf5::types::{VarLenArray, VarLenUnicode};
    use std::str::FromStr;

    #[derive(hdf5::H5Type, Clone, Debug)]
    #[repr(C)]
    struct Row {
        label: VarLenUnicode,
        id: i64,
        samples: VarLenArray<i32>,
    }

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_compound_two_vlen.h5");
    let dst = dir.path().join("compound_two_vlen_repacked.h5");

    let labels = ["first", "", "third"];
    let samples: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![], vec![-9, 400_000]];
    {
        let file = hdf5::File::create(&src).unwrap();
        let vals: Vec<Row> = labels
            .iter()
            .zip(&samples)
            .enumerate()
            .map(|(i, (l, s))| Row {
                label: VarLenUnicode::from_str(l).unwrap(),
                id: i as i64 - 1,
                samples: VarLenArray::from_slice(s.as_slice()),
            })
            .collect();
        file.new_dataset::<Row>()
            .shape((labels.len(),))
            .create("rows")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let c = hdf5::File::open(&dst).unwrap();
    let cvals = c.dataset("rows").unwrap().read_raw::<Row>().unwrap();
    assert_eq!(cvals.len(), labels.len());
    for i in 0..labels.len() {
        assert_eq!(cvals[i].label.as_str(), labels[i], "row {i} label differs");
        assert_eq!(cvals[i].id, i as i64 - 1, "row {i} id differs");
        assert_eq!(
            cvals[i].samples.as_slice(),
            samples[i].as_slice(),
            "row {i} samples differ"
        );
    }
}

/// A chunked compound with a variable-length member has to reach the re-staging
/// path rather than the verbatim chunk copy: copying compressed chunks would
/// carry the source's heap addresses through untouched, and the destination's own
/// collections have to be placed before the chunks are encoded (issue #109).
#[test]
fn repack_roundtrips_c_chunked_compound_with_vlen_member() {
    use hdf5::types::VarLenUnicode;
    use std::str::FromStr;

    #[derive(hdf5::H5Type, Clone, Debug)]
    #[repr(C)]
    struct Row {
        id: i32,
        label: VarLenUnicode,
    }

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_chunked_compound_vlen.h5");
    let dst = dir.path().join("chunked_compound_vlen_repacked.h5");

    let n = 40usize;
    let labels: Vec<String> = (0..n).map(|i| format!("row-{i}")).collect();
    {
        let file = hdf5::File::create(&src).unwrap();
        let vals: Vec<Row> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| Row {
                id: i as i32,
                label: VarLenUnicode::from_str(l).unwrap(),
            })
            .collect();
        file.new_dataset::<Row>()
            .shape((n,))
            .chunk((8,))
            .create("rows")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    // Chunking must survive the rebuild, not be flattened to contiguous.
    let f = File::open(&dst).unwrap();
    assert!(
        matches!(
            f.dataset("rows").unwrap().layout().unwrap(),
            hdf5_pure::Layout::Chunked { .. }
        ),
        "repacked dataset must stay chunked"
    );

    let c = hdf5::File::open(&dst).unwrap();
    let cvals = c.dataset("rows").unwrap().read_raw::<Row>().unwrap();
    assert_eq!(cvals.len(), n);
    for i in 0..n {
        assert_eq!(cvals[i].id, i as i32, "row {i} id differs");
        assert_eq!(cvals[i].label.as_str(), labels[i], "row {i} label differs");
    }
}

/// An object-header address embedded in a compound member goes stale on rewrite
/// for the same reason a top-level one does: the destination puts the target
/// object somewhere else. Copying the element bytes verbatim leaves a file whose
/// fixed-size members read back correctly and whose references dereference into
/// whatever now occupies that address — which is why this test dereferences
/// rather than merely reading the rows.
#[test]
fn repack_roundtrips_c_compound_with_object_reference_member() {
    use hdf5::{ObjectReference, ObjectReference1, ReferencedObject};

    #[derive(hdf5::H5Type, Clone, Debug)]
    #[repr(C)]
    struct Row {
        id: i32,
        target: ObjectReference1,
    }

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_compound_ref.h5");
    let dst = dir.path().join("compound_ref_repacked.h5");

    {
        let file = hdf5::File::create(&src).unwrap();
        file.new_dataset::<i32>()
            .shape((3,))
            .create("alpha")
            .unwrap()
            .write(&[1i32, 2, 3])
            .unwrap();
        let grp = file.create_group("grp").unwrap();
        grp.new_dataset::<i32>()
            .shape((2,))
            .create("beta")
            .unwrap()
            .write(&[40i32, 50])
            .unwrap();
        let vals = vec![
            Row {
                id: 7,
                target: ObjectReference1::create(&file, "alpha").unwrap(),
            },
            Row {
                id: 8,
                target: ObjectReference1::create(&file, "grp/beta").unwrap(),
            },
        ];
        file.new_dataset::<Row>()
            .shape((vals.len(),))
            .create("rows")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let c = hdf5::File::open(&dst).unwrap();
    let cvals = c.dataset("rows").unwrap().read_raw::<Row>().unwrap();
    assert_eq!(cvals.len(), 2);
    assert_eq!(cvals[0].id, 7);
    assert_eq!(cvals[1].id, 8);

    // Each embedded reference must resolve to its own target's *new* address, and
    // that target must still hold the right data.
    let expected: [&[i32]; 2] = [&[1, 2, 3], &[40, 50]];
    for (i, row) in cvals.iter().enumerate() {
        match row.target.dereference(&c).unwrap() {
            ReferencedObject::Dataset(ds) => assert_eq!(
                ds.read_raw::<i32>().unwrap(),
                expected[i],
                "row {i} reference resolves to the wrong dataset"
            ),
            other => panic!("row {i}: expected a dataset, got {other:?}"),
        }
    }
}

/// The refusals that guard the top-level object-reference path have to guard the
/// embedded one too: a reference into a dropped subtree cannot be rewritten to
/// anything meaningful, so the repack must fail by name rather than emit a
/// dangling address.
#[test]
fn repack_refuses_compound_reference_to_dropped_object() {
    use hdf5::{ObjectReference, ObjectReference1};

    #[derive(hdf5::H5Type, Clone, Debug)]
    #[repr(C)]
    struct Row {
        id: i32,
        target: ObjectReference1,
    }

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_compound_ref_dropped.h5");
    let dst = dir.path().join("compound_ref_dropped_repacked.h5");

    {
        let file = hdf5::File::create(&src).unwrap();
        file.new_dataset::<i32>()
            .shape((2,))
            .create("doomed")
            .unwrap()
            .write(&[1i32, 2])
            .unwrap();
        let vals = vec![Row {
            id: 1,
            target: ObjectReference1::create(&file, "doomed").unwrap(),
        }];
        file.new_dataset::<Row>()
            .shape((1,))
            .create("rows")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    let err = repack(&src, &dst, &RepackOptions::new().drop_path("doomed")).unwrap_err();
    match err {
        hdf5_pure::Error::RepackUnsupported(msg) => assert!(
            msg.contains("rows") && msg.contains("doomed"),
            "error should name the dataset and its dropped target: {msg}"
        ),
        other => panic!("expected RepackUnsupported, got {other:?}"),
    }
    assert!(!dst.exists(), "dst must not be created when repack refuses");
}

/// A compound can carry a variable-length member *and* an object-reference
/// member. Both kinds of embedded address go stale on rewrite, so rewriting only
/// the first kind found leaves the other pointing into the source file — the
/// original defect surviving for the one shape that reaches both paths.
///
/// The reference is dereferenced rather than merely read back, because a stale
/// address reads as a perfectly ordinary 8 bytes.
#[test]
fn repack_roundtrips_c_compound_with_both_vlen_and_reference_members() {
    use hdf5::types::VarLenUnicode;
    use hdf5::{ObjectReference, ObjectReference1, ReferencedObject};
    use std::str::FromStr;

    #[derive(hdf5::H5Type, Clone, Debug)]
    #[repr(C)]
    struct Row {
        label: VarLenUnicode,
        id: i32,
        target: ObjectReference1,
    }

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_compound_both.h5");
    let dst = dir.path().join("compound_both_repacked.h5");

    let labels = ["alpha", "", "gamma"];
    {
        let file = hdf5::File::create(&src).unwrap();
        file.new_dataset::<i32>()
            .shape((3,))
            .create("pointee")
            .unwrap()
            .write(&[11i32, 22, 33])
            .unwrap();
        let vals: Vec<Row> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| Row {
                label: VarLenUnicode::from_str(l).unwrap(),
                id: i as i32,
                target: ObjectReference1::create(&file, "pointee").unwrap(),
            })
            .collect();
        file.new_dataset::<Row>()
            .shape((labels.len(),))
            .create("rows")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let c = hdf5::File::open(&dst).unwrap();
    let cvals = c.dataset("rows").unwrap().read_raw::<Row>().unwrap();
    assert_eq!(cvals.len(), labels.len());
    for (i, row) in cvals.iter().enumerate() {
        assert_eq!(row.label.as_str(), labels[i], "row {i} label differs");
        assert_eq!(row.id, i as i32, "row {i} id differs");
        match row.target.dereference(&c).unwrap() {
            ReferencedObject::Dataset(ds) => assert_eq!(
                ds.read_raw::<i32>().unwrap(),
                vec![11, 22, 33],
                "row {i} reference resolves to the wrong data"
            ),
            other => panic!("row {i}: expected a dataset, got {other:?}"),
        }
    }
}

/// The object-reference refusals must not be reachable-around by adding a
/// variable-length member: a reference into a dropped subtree cannot be rewritten
/// to anything meaningful whatever else the compound carries.
#[test]
fn repack_refuses_dropped_target_in_a_compound_that_also_has_a_vlen_member() {
    use hdf5::types::VarLenUnicode;
    use hdf5::{ObjectReference, ObjectReference1};
    use std::str::FromStr;

    #[derive(hdf5::H5Type, Clone, Debug)]
    #[repr(C)]
    struct Row {
        label: VarLenUnicode,
        target: ObjectReference1,
    }

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_both_dropped.h5");
    let dst = dir.path().join("both_dropped_repacked.h5");

    {
        let file = hdf5::File::create(&src).unwrap();
        file.new_dataset::<i32>()
            .shape((2,))
            .create("doomed")
            .unwrap()
            .write(&[1i32, 2])
            .unwrap();
        let vals = vec![Row {
            label: VarLenUnicode::from_str("x").unwrap(),
            target: ObjectReference1::create(&file, "doomed").unwrap(),
        }];
        file.new_dataset::<Row>()
            .shape((1,))
            .create("rows")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    let err = repack(&src, &dst, &RepackOptions::new().drop_path("doomed")).unwrap_err();
    match err {
        hdf5_pure::Error::RepackUnsupported(msg) => assert!(
            msg.contains("rows") && msg.contains("doomed"),
            "error should name the dataset and its dropped target: {msg}"
        ),
        other => panic!("expected RepackUnsupported, got {other:?}"),
    }
    assert!(!dst.exists(), "dst must not be created when repack refuses");
}

/// Likewise the chunked refusal: an object address inside a compressed chunk
/// would need rewriting in place, and a variable-length member alongside it does
/// not make that possible.
#[test]
fn repack_refuses_chunked_compound_with_both_vlen_and_reference_members() {
    use hdf5::types::VarLenUnicode;
    use hdf5::{ObjectReference, ObjectReference1};
    use std::str::FromStr;

    #[derive(hdf5::H5Type, Clone, Debug)]
    #[repr(C)]
    struct Row {
        label: VarLenUnicode,
        target: ObjectReference1,
    }

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_both_chunked.h5");
    let dst = dir.path().join("both_chunked_repacked.h5");

    {
        let file = hdf5::File::create(&src).unwrap();
        file.new_dataset::<i32>()
            .shape((2,))
            .create("pointee")
            .unwrap()
            .write(&[1i32, 2])
            .unwrap();
        let vals: Vec<Row> = (0..8)
            .map(|i| Row {
                label: VarLenUnicode::from_str(&format!("r{i}")).unwrap(),
                target: ObjectReference1::create(&file, "pointee").unwrap(),
            })
            .collect();
        file.new_dataset::<Row>()
            .shape((8,))
            .chunk((4,))
            .create("rows")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    let err = repack(&src, &dst, &RepackOptions::new()).unwrap_err();
    match err {
        hdf5_pure::Error::RepackUnsupported(msg) => assert!(
            msg.contains("rows") && msg.contains("object-reference"),
            "error should name the dataset and the reason: {msg}"
        ),
        other => panic!("expected RepackUnsupported, got {other:?}"),
    }
}

/// The attribute encodings the reference C library can write and this crate
/// cannot, carried across a repack and read back by that same library
/// (issue #241).
///
/// This is the half of the fidelity contract no pure-Rust test can reach.
/// `AttrValue` has no narrow integer array, no variable-length string, and no
/// rank above one, so a source this crate *writes* cannot exhibit those losses
/// at all — only a C-written file can, which is exactly the file a user repacks.
/// Every assertion is made through the C library's own type system, so it states
/// what a consumer sees rather than what this crate's reader reports.
#[test]
fn c_written_attribute_encodings_survive_a_repack() {
    use hdf5::types::{FixedAscii, FloatSize, IntSize, TypeDescriptor, VarLenUnicode};
    use std::str::FromStr;

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_attr_src.h5");
    let dst = dir.path().join("c_attr_dst.h5");

    {
        let file = hdf5::File::create(&src).unwrap();
        let ds = file
            .new_dataset::<f64>()
            .shape((4,))
            .create("data")
            .unwrap();
        ds.write(&[1.0f64, 2.0, 3.0, 4.0]).unwrap();

        // One writer for both owners: an attribute on the root group takes a
        // different path through the repack walk than one on a dataset, and the
        // widening hit both.
        let (ds_loc, root_loc): (&hdf5::Location, &hdf5::Location) = (&ds, &file);
        for owner in [ds_loc, root_loc] {
            owner
                .new_attr::<i8>()
                .create("i8")
                .unwrap()
                .write_scalar(&-3i8)
                .unwrap();
            owner
                .new_attr::<i32>()
                .create("i32")
                .unwrap()
                .write_scalar(&-7i32)
                .unwrap();
            owner
                .new_attr::<u16>()
                .create("u16")
                .unwrap()
                .write_scalar(&65535u16)
                .unwrap();
            owner
                .new_attr::<f32>()
                .create("f32")
                .unwrap()
                .write_scalar(&1.5f32)
                .unwrap();
            owner
                .new_attr::<i16>()
                .shape([3])
                .create("i16arr")
                .unwrap()
                .write(&[1i16, 2, 3])
                .unwrap();
            // Rank 2: `AttrValue`'s array variants are all one-dimensional, so a
            // decode flattens this to six elements.
            owner
                .new_attr::<i32>()
                .shape([2, 3])
                .create("rank2")
                .unwrap()
                .write_raw(&[1i32, 2, 3, 4, 5, 6])
                .unwrap();
            // A true variable-length string, which this crate's writer never
            // emits and a decode turns into a fixed-width one.
            owner
                .new_attr::<VarLenUnicode>()
                .create("vlstr")
                .unwrap()
                .write_scalar(&VarLenUnicode::from_str("hello").unwrap())
                .unwrap();
            // The same, as an array: a scalar and a one-element-per-entry array
            // reach the writer by different paths.
            owner
                .new_attr::<VarLenUnicode>()
                .shape([2])
                .create("vlstrs")
                .unwrap()
                .write_raw(&[
                    VarLenUnicode::from_str("alpha").unwrap(),
                    VarLenUnicode::from_str("beta").unwrap(),
                ])
                .unwrap();
            // A fixed-width string declared far wider than its content: the
            // declared width is the part a decode drops, since it reports the
            // content and nothing else.
            owner
                .new_attr::<FixedAscii<16>>()
                .create("units")
                .unwrap()
                .write_scalar(&FixedAscii::<16>::from_ascii("m/s").unwrap())
                .unwrap();
            // Rank 2 over strings rather than numbers, which is the shape
            // issue #241 measured being flattened.
            owner
                .new_attr::<FixedAscii<4>>()
                .shape([2, 2])
                .create("grid")
                .unwrap()
                .write_raw(&[
                    FixedAscii::<4>::from_ascii("ab").unwrap(),
                    FixedAscii::<4>::from_ascii("cd").unwrap(),
                    FixedAscii::<4>::from_ascii("ef").unwrap(),
                    FixedAscii::<4>::from_ascii("gh").unwrap(),
                ])
                .unwrap();
        }
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    // One row per entry in issue #241's table of what a decode loses.
    let expected: [(&str, TypeDescriptor, Vec<usize>); 10] = [
        ("i8", TypeDescriptor::Integer(IntSize::U1), vec![]),
        ("i32", TypeDescriptor::Integer(IntSize::U4), vec![]),
        ("u16", TypeDescriptor::Unsigned(IntSize::U2), vec![]),
        ("f32", TypeDescriptor::Float(FloatSize::U4), vec![]),
        ("i16arr", TypeDescriptor::Integer(IntSize::U2), vec![3]),
        ("rank2", TypeDescriptor::Integer(IntSize::U4), vec![2, 3]),
        ("vlstr", TypeDescriptor::VarLenUnicode, vec![]),
        ("vlstrs", TypeDescriptor::VarLenUnicode, vec![2]),
        // 16, not 3: the declared width outlives the content.
        ("units", TypeDescriptor::FixedAscii(16), vec![]),
        ("grid", TypeDescriptor::FixedAscii(4), vec![2, 2]),
    ];

    let c = hdf5::File::open(&dst).unwrap();
    let ds = c.dataset("data").unwrap();
    let (ds_loc, root_loc): (&hdf5::Location, &hdf5::Location) = (&ds, &c);
    for (owner_name, owner) in [("dataset data", ds_loc), ("root group", root_loc)] {
        for (name, descriptor, shape) in &expected {
            let attr = owner
                .attr(name)
                .unwrap_or_else(|e| panic!("{owner_name} lost attribute {name:?}: {e}"));
            assert_eq!(
                &attr.dtype().unwrap().to_descriptor().unwrap(),
                descriptor,
                "{owner_name}: attribute {name:?} changed datatype across the repack"
            );
            assert_eq!(
                &attr.shape(),
                shape,
                "{owner_name}: attribute {name:?} changed shape across the repack"
            );
        }
        // The variable-length string is the one whose bytes cannot be copied, so
        // it proves the fallback re-encoded it against the destination's own heap
        // rather than leaving an address pointing into the source.
        assert_eq!(
            owner
                .attr("vlstr")
                .unwrap()
                .read_scalar::<VarLenUnicode>()
                .unwrap()
                .as_str(),
            "hello",
            "{owner_name}: the variable-length string must still resolve"
        );
        assert_eq!(
            owner.attr("rank2").unwrap().read_raw::<i32>().unwrap(),
            vec![1, 2, 3, 4, 5, 6]
        );
        assert_eq!(
            owner
                .attr("units")
                .unwrap()
                .read_scalar::<FixedAscii<16>>()
                .unwrap()
                .as_str(),
            "m/s"
        );
        assert_eq!(
            owner
                .attr("vlstrs")
                .unwrap()
                .read_raw::<VarLenUnicode>()
                .unwrap()
                .iter()
                .map(|s| s.as_str().to_owned())
                .collect::<Vec<_>>(),
            ["alpha", "beta"]
        );
        assert_eq!(owner.attr("i8").unwrap().read_scalar::<i8>().unwrap(), -3);
        assert_eq!(
            owner.attr("f32").unwrap().read_scalar::<f32>().unwrap(),
            1.5
        );
    }
}
