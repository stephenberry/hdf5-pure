//! In-place editing of files that carry a userblock (non-zero base address) —
//! the userblock slice of issue #104.
//!
//! Every MATLAB v7.3 `.mat` file begins with a 512-byte userblock, so the HDF5
//! superblock sits at byte 512 and every stored address is relative to that
//! base. These tests exercise the editor on such files — a synthetic one built
//! by this crate and a real `.mat` fixture — and verify that the userblock bytes
//! survive an edit byte-for-byte.
//!
//! This slice supports contiguous value overwrites, additions of contiguous and
//! chunked/filtered datasets, overwrites of chunked datasets (in place when the
//! re-encoded chunks fit their slots, otherwise rebuilt and relocated with the
//! old storage reclaimed), group creation, and compact group attributes.
//! Deletions, copies, and relocating overwrites of *contiguous* datasets are
//! refused cleanly on a userblock file (covered below); they remain follow-up
//! work, and a refusal never corrupts the file.

use hdf5_pure::{AttrValue, EditSession, File, FileBuilder};

const UB: usize = 512;
const MARKER: &[u8] = b"USERBLOCK-MARKER-0104";

/// Build a userblock file with two root datasets and a nested group+dataset,
/// stamping a recognizable marker across the userblock region. Returns the
/// 512-byte userblock as written, for later byte-for-byte comparison.
fn build_userblock_file(path: &std::path::Path) -> Vec<u8> {
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("alpha").with_f64_data(&[1.0, 2.0, 3.0, 4.0]);
    b.create_dataset("beta").with_i32_data(&[10, 20, 30]);
    let mut g = b.create_group("grp");
    g.create_dataset("inner").with_f64_data(&[7.5, 8.5]);
    b.add_group(g.finish());
    let mut bytes = b.finish().unwrap();

    // The userblock region [0..512] is zero-filled by the writer; stamp a marker
    // at the start and a sentinel at the last byte to catch any stray write into
    // the userblock during an edit.
    bytes[..MARKER.len()].copy_from_slice(MARKER);
    bytes[UB - 1] = 0xAB;
    let userblock = bytes[..UB].to_vec();
    std::fs::write(path, &bytes).unwrap();
    userblock
}

#[test]
fn synthetic_userblock_file_roundtrip() {
    let path = std::env::temp_dir().join("hdf5_pure_ub_roundtrip.h5");
    let userblock = build_userblock_file(&path);

    {
        let mut s = EditSession::open(&path).unwrap();
        // Same-length in-place overwrite of an existing contiguous dataset.
        s.write_dataset("alpha").with_f64_data(&[9.0, 8.0, 7.0, 6.0]);
        // Add contiguous datasets at the root and inside the nested group.
        s.create_dataset("added").with_i32_data(&[100, 200]);
        s.create_dataset("grp/added_inner").with_f64_data(&[3.25]);
        // Create a new group and set a compact attribute on it.
        s.create_group("newgrp");
        s.set_group_attr("newgrp", "tag", AttrValue::AsciiString("v104".into()));
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    // Overwritten value.
    assert_eq!(
        file.dataset("alpha").unwrap().read_f64().unwrap(),
        vec![9.0, 8.0, 7.0, 6.0]
    );
    // Untouched originals.
    assert_eq!(
        file.dataset("beta").unwrap().read_i32().unwrap(),
        vec![10, 20, 30]
    );
    assert_eq!(
        file.dataset("grp/inner").unwrap().read_f64().unwrap(),
        vec![7.5, 8.5]
    );
    // Additions.
    assert_eq!(
        file.dataset("added").unwrap().read_i32().unwrap(),
        vec![100, 200]
    );
    assert_eq!(
        file.dataset("grp/added_inner").unwrap().read_f64().unwrap(),
        vec![3.25]
    );
    // New group and its attribute.
    let mut groups = file.root().groups().unwrap();
    groups.sort();
    assert!(groups.contains(&"newgrp".to_string()));
    let attrs = file.group("newgrp").unwrap().attrs().unwrap();
    assert_eq!(attrs.get("tag"), Some(&AttrValue::String("v104".into())));

    // The userblock bytes are preserved verbatim across the edit.
    let after = std::fs::read(&path).unwrap();
    assert_eq!(
        &after[..UB],
        &userblock[..],
        "userblock bytes changed across the edit"
    );
    assert_eq!(&after[..MARKER.len()], MARKER);
    assert_eq!(after[UB - 1], 0xAB);

    std::fs::remove_file(&path).ok();
}

#[test]
fn userblock_inplace_overwrite_only_takes_fast_path() {
    // A lone same-length overwrite takes the in-place fast path (no header
    // rewrite, no superblock flip); it must work on a userblock file and leave
    // the userblock untouched.
    let path = std::env::temp_dir().join("hdf5_pure_ub_inplace_only.h5");
    let userblock = build_userblock_file(&path);

    {
        let mut s = EditSession::open(&path).unwrap();
        s.write_dataset("beta").with_i32_data(&[-1, -2, -3]);
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("beta").unwrap().read_i32().unwrap(),
        vec![-1, -2, -3]
    );
    assert_eq!(
        file.dataset("alpha").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(&std::fs::read(&path).unwrap()[..UB], &userblock[..]);

    std::fs::remove_file(&path).ok();
}

/// Each unsupported operation on a userblock file must be refused with
/// `EditUnsupported`, and the file must remain valid with its original contents.
#[test]
fn userblock_unsupported_ops_are_refused_cleanly() {
    // deletion
    {
        let path = std::env::temp_dir().join("hdf5_pure_ub_refuse_delete.h5");
        build_userblock_file(&path);
        let mut s = EditSession::open(&path).unwrap();
        s.delete("alpha");
        assert!(s.commit().is_err(), "delete should be refused");
        drop(s);
        let file = File::open(&path).unwrap();
        assert_eq!(
            file.dataset("alpha").unwrap().read_f64().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        std::fs::remove_file(&path).ok();
    }

    // copy
    {
        let path = std::env::temp_dir().join("hdf5_pure_ub_refuse_copy.h5");
        build_userblock_file(&path);
        let mut s = EditSession::open(&path).unwrap();
        s.copy("alpha", "alpha_copy");
        assert!(s.commit().is_err(), "copy should be refused");
        drop(s);
        let file = File::open(&path).unwrap();
        assert!(file.dataset("alpha_copy").is_err());
        assert_eq!(
            file.dataset("alpha").unwrap().read_f64().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        std::fs::remove_file(&path).ok();
    }

    // relocating overwrite of a *contiguous* dataset (a resize). The refusal
    // happens in the write preflight before any byte is written, so the file must
    // be byte-identical afterward and the original data must still read back.
    {
        let path = std::env::temp_dir().join("hdf5_pure_ub_refuse_contig_resize.h5");
        build_userblock_file(&path);
        let before = std::fs::read(&path).unwrap();

        let mut s = EditSession::open(&path).unwrap();
        // A longer value than the on-disk extent forces a relocation.
        s.write_dataset("alpha").with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(
            s.commit().is_err(),
            "contiguous resizing overwrite should be refused"
        );
        drop(s);
        assert_eq!(
            std::fs::read(&path).unwrap(),
            before,
            "a refused resizing overwrite must not modify the file"
        );
        assert_eq!(
            File::open(&path).unwrap().dataset("alpha").unwrap().read_f64().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        std::fs::remove_file(&path).ok();
    }
}

#[test]
fn real_mat_add_dataset_preserves_userblock_and_data() {
    // Copy the fixture so the test never mutates the checked-in file.
    let src = std::path::Path::new("tests/fixtures/mat_real/test_string_v73.mat");
    let path = std::env::temp_dir().join("hdf5_pure_ub_real_mat.mat");
    std::fs::copy(src, &path).unwrap();

    let original_userblock = std::fs::read(&path).unwrap()[..UB].to_vec();
    assert_eq!(
        &original_userblock[..10],
        b"MATLAB 7.3",
        "fixture is not a MATLAB v7.3 userblock file"
    );

    // Capture an existing variable's bytes before editing.
    let before = File::open(&path)
        .unwrap()
        .dataset("string_scalar")
        .unwrap()
        .read_u32()
        .unwrap();

    {
        let mut s = EditSession::open(&path).unwrap();
        s.create_dataset("hdf5_pure_probe")
            .with_f64_data(&[42.0, -1.0, 3.5]);
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    // The added dataset reads back correctly.
    assert_eq!(
        file.dataset("hdf5_pure_probe").unwrap().read_f64().unwrap(),
        vec![42.0, -1.0, 3.5]
    );
    // The untouched original still reads identically.
    assert_eq!(
        file.dataset("string_scalar").unwrap().read_u32().unwrap(),
        before
    );
    // The original variables and MATLAB's MCOS groups are still present.
    let datasets = file.root().datasets().unwrap();
    for name in ["string_array", "string_empty", "string_scalar", "hdf5_pure_probe"] {
        assert!(datasets.contains(&name.to_string()), "missing dataset {name}");
    }
    let groups = file.root().groups().unwrap();
    assert!(groups.contains(&"#refs#".to_string()));
    assert!(groups.contains(&"#subsystem#".to_string()));

    // The MATLAB userblock (signature, version, endian indicator) is unchanged.
    let after_userblock = std::fs::read(&path).unwrap()[..UB].to_vec();
    assert_eq!(
        after_userblock, original_userblock,
        "MATLAB userblock changed across the edit"
    );

    std::fs::remove_file(&path).ok();
}
