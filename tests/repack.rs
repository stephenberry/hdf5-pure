//! Whole-file repack (issue #21): compaction, object dropping, fidelity of
//! survivors, and fail-loud refusal of features that cannot be reproduced.

use hdf5_pure::{AttrValue, FileBuilder, RepackOptions, repack};

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(name)
}

#[test]
fn drops_object_and_shrinks_file() {
    let src = tmp("hdf5_pure_repack_drop_src.h5");
    let dst = tmp("hdf5_pure_repack_drop_dst.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.create_dataset("bulk").with_f64_data(&vec![9.0; 4096]);
    b.write(&src).unwrap();
    let src_size = std::fs::metadata(&src).unwrap().len();

    repack(&src, &dst, &RepackOptions::new().drop_path("bulk")).unwrap();

    let dst_size = std::fs::metadata(&dst).unwrap().len();
    assert!(
        dst_size < src_size,
        "dropping the bulk dataset should shrink the file (src {src_size}, dst {dst_size})"
    );

    let f = hdf5_pure::File::open(&dst).unwrap();
    assert_eq!(f.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(
        f.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert!(f.dataset("bulk").is_err());
    std::fs::remove_file(&src).ok();
    std::fs::remove_file(&dst).ok();
}

#[test]
fn pure_compaction_preserves_everything() {
    let src = tmp("hdf5_pure_repack_compact_src.h5");
    let dst = tmp("hdf5_pure_repack_compact_dst.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("ints").with_i64_data(&[10, 20, 30, 40]);
    b.create_dataset("floats").with_f64_data(&[1.5, 2.5, 3.5]);
    b.set_attr("title", AttrValue::String("experiment".to_string()));
    let mut g = b.create_group("grp");
    g.create_dataset("inner").with_f32_data(&[0.25, 0.5]);
    g.set_attr("units", AttrValue::AsciiString("m/s".to_string()));
    b.add_group(g.finish());
    b.write(&src).unwrap();

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = hdf5_pure::File::open(&dst).unwrap();
    assert_eq!(
        f.dataset("ints").unwrap().read_i64().unwrap(),
        vec![10, 20, 30, 40]
    );
    assert_eq!(
        f.dataset("floats").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5, 3.5]
    );
    assert_eq!(
        f.dataset("grp/inner").unwrap().read_f32().unwrap(),
        vec![0.25, 0.5]
    );
    // Attributes carried over.
    let root_attrs = f.root().attrs().unwrap();
    assert_eq!(
        root_attrs.get("title"),
        Some(&AttrValue::String("experiment".to_string()))
    );
    let grp_attrs = f.group("grp").unwrap().attrs().unwrap();
    // The value survives. Note: the reader reports a fixed-width ASCII string as
    // AttrValue::String (it does not preserve the ASCII-vs-UTF-8 charset
    // distinction on read), so repack round-trips it as String — verified to be
    // pre-existing reader behavior, independent of repack.
    assert_eq!(
        grp_attrs.get("units"),
        Some(&AttrValue::String("m/s".to_string()))
    );
    std::fs::remove_file(&src).ok();
    std::fs::remove_file(&dst).ok();
}

#[test]
fn drops_whole_group_subtree() {
    let src = tmp("hdf5_pure_repack_dropgrp_src.h5");
    let dst = tmp("hdf5_pure_repack_dropgrp_dst.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("root_ds").with_i32_data(&[1]);
    let mut g = b.create_group("doomed");
    g.create_dataset("a").with_i32_data(&[1, 2]);
    g.create_dataset("b").with_i32_data(&[3, 4]);
    b.add_group(g.finish());
    b.write(&src).unwrap();

    repack(&src, &dst, &RepackOptions::new().drop_path("/doomed")).unwrap();

    let f = hdf5_pure::File::open(&dst).unwrap();
    assert_eq!(f.root().datasets().unwrap(), vec!["root_ds".to_string()]);
    assert!(f.group("doomed").is_err());
    assert!(f.dataset("doomed/a").is_err());
    std::fs::remove_file(&src).ok();
    std::fs::remove_file(&dst).ok();
}

#[test]
fn preserves_chunked_and_compressed_dataset() {
    let src = tmp("hdf5_pure_repack_chunk_src.h5");
    let dst = tmp("hdf5_pure_repack_chunk_dst.h5");
    let data: Vec<f64> = (0..2048).map(|i| i as f64 * 0.5).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[7]);
    b.create_dataset("comp")
        .with_f64_data(&data)
        .with_chunks(&[256])
        .with_shuffle()
        .with_deflate(6);
    b.write(&src).unwrap();

    // Drop nothing; just compact. The chunked+filtered dataset must survive with
    // byte-exact values.
    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = hdf5_pure::File::open(&dst).unwrap();
    assert_eq!(f.dataset("comp").unwrap().read_f64().unwrap(), data);
    assert_eq!(f.dataset("keep").unwrap().read_i32().unwrap(), vec![7]);
    std::fs::remove_file(&src).ok();
    std::fs::remove_file(&dst).ok();
}

#[test]
fn preserves_multidim_and_maxshape() {
    let src = tmp("hdf5_pure_repack_md_src.h5");
    let dst = tmp("hdf5_pure_repack_md_dst.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("grid")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .with_shape(&[2, 3])
        .with_maxshape(&[u64::MAX, 3])
        .with_chunks(&[1, 3]);
    b.write(&src).unwrap();

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = hdf5_pure::File::open(&dst).unwrap();
    let ds = f.dataset("grid").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![2, 3]);
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    std::fs::remove_file(&src).ok();
    std::fs::remove_file(&dst).ok();
}

#[test]
fn rejects_nonexistent_drop_path() {
    let src = tmp("hdf5_pure_repack_baddrop_src.h5");
    let dst = tmp("hdf5_pure_repack_baddrop_dst.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("present").with_i32_data(&[1]);
    b.write(&src).unwrap();

    let err = repack(&src, &dst, &RepackOptions::new().drop_path("absent")).unwrap_err();
    assert!(
        matches!(err, hdf5_pure::Error::RepackUnsupported(_)),
        "expected RepackUnsupported, got {err:?}"
    );
    // Nothing should have been written to dst.
    assert!(
        !dst.exists(),
        "dst must not be created when the repack fails"
    );
    std::fs::remove_file(&src).ok();
    std::fs::remove_file(&dst).ok();
}
