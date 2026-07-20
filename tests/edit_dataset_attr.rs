//! Pure-Rust tests for `EditSession::set_dataset_attr` / `remove_dataset_attr`
//! (issue #146): compact dataset-attribute add / update / remove, staged and
//! applied on commit by relocating the dataset's object header while preserving its
//! data and chunk index. C-library interop — undefined-`AttributeInfo` acceptance
//! (dataset and group), the dense-storage refusal, and the single-hard-link refusal
//! — lives in `edit_crosscheck.rs`.

#![allow(deprecated)] // exercises the deprecated EditSession/SwmrWriter shims (issue #148)
use hdf5_pure::{AttrValue, EditSession, Error, File, FileBuilder};
use tempfile::tempdir;

fn build_contig(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3, 4]);
    b.write(path).unwrap();
}

fn build_chunked(path: &std::path::Path, n: i32, chunk: u64) {
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&(0..n).collect::<Vec<_>>())
        .with_shape(&[n as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[chunk]);
    b.write(path).unwrap();
}

#[test]
fn set_fixed_attrs_preserves_data() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build_contig(&p);

    {
        let mut s = EditSession::open(&p).unwrap();
        s.set_dataset_attr("d", "count", AttrValue::I64(42));
        s.set_dataset_attr("d", "unit", AttrValue::String("m/s".into()));
        s.commit().unwrap();
    }

    let f = File::open(&p).unwrap();
    let d = f.dataset("d").unwrap();
    assert_eq!(d.read_i32().unwrap(), vec![1, 2, 3, 4]); // data untouched
    let attrs = d.attrs().unwrap();
    assert_eq!(attrs.get("count"), Some(&AttrValue::I64(42)));
    assert_eq!(attrs.get("unit"), Some(&AttrValue::String("m/s".into())));
}

#[test]
fn update_and_remove_attr() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build_contig(&p);

    {
        let mut s = EditSession::open(&p).unwrap();
        s.set_dataset_attr("d", "a", AttrValue::I64(1));
        s.set_dataset_attr("d", "b", AttrValue::I64(2));
        s.commit().unwrap();
    }
    {
        let mut s = EditSession::open(&p).unwrap();
        s.set_dataset_attr("d", "a", AttrValue::I64(99)); // replace
        s.remove_dataset_attr("d", "b"); // remove
        s.commit().unwrap();
    }

    let f = File::open(&p).unwrap();
    let attrs = f.dataset("d").unwrap().attrs().unwrap();
    assert_eq!(attrs.get("a"), Some(&AttrValue::I64(99)));
    assert!(!attrs.contains_key("b"));
}

#[test]
fn set_vlstring_attr() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build_contig(&p);

    {
        let mut s = EditSession::open(&p).unwrap();
        s.set_dataset_attr(
            "d",
            "labels",
            AttrValue::VarLenAsciiArray(vec!["alpha".into(), "beta".into()]),
        );
        s.commit().unwrap();
    }

    let f = File::open(&p).unwrap();
    let attrs = f.dataset("d").unwrap().attrs().unwrap();
    assert_eq!(
        attrs.get("labels"),
        Some(&AttrValue::StringArray(vec!["alpha".into(), "beta".into()]))
    );
}

#[test]
fn attr_edit_on_chunked_preserves_data_and_index() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build_chunked(&p, 8, 4);

    {
        let mut s = EditSession::open(&p).unwrap();
        // Immediate append grows the Extensible Array in place.
        s.append_inplace_i32("d", &[8, 9, 10, 11]).unwrap(); // 0..12
        // A staged attribute edit relocates the header at commit (preserving the
        // grown index and data). The cache is invalidated at commit entry.
        s.set_dataset_attr("d", "tag", AttrValue::I64(5));
        s.commit().unwrap();
        // Append again: re-locates at the dataset's new header address.
        s.append_inplace_i32("d", &[12, 13]).unwrap(); // 0..14
    }

    let f = File::open(&p).unwrap();
    let d = f.dataset("d").unwrap();
    assert_eq!(d.read_i32().unwrap(), (0..14).collect::<Vec<_>>());
    assert_eq!(d.attrs().unwrap().get("tag"), Some(&AttrValue::I64(5)));
}

#[test]
fn guard_refuses_append_after_staged_dataset_attr() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build_chunked(&p, 4, 4);

    let mut s = EditSession::open(&p).unwrap();
    s.append_inplace_i32("d", &[4, 5, 6, 7]).unwrap(); // fine, nothing staged
    s.set_dataset_attr("d", "tag", AttrValue::I64(1)); // now stage an attr edit
    // A second in-place append is refused: commit would relocate the header this
    // append planned against.
    let err = s.append_inplace_i32("d", &[8]).unwrap_err();
    assert!(matches!(err, Error::AppendInPlaceUnsupported(_)));
}

#[test]
fn attr_edit_plus_append_dataset_same_commit_refused() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build_chunked(&p, 8, 4);

    let mut s = EditSession::open(&p).unwrap();
    s.set_dataset_attr("d", "tag", AttrValue::I64(1));
    s.append_dataset("d").append_i32(&[8, 9, 10, 11]);
    let err = s.commit().unwrap_err();
    assert!(matches!(err, Error::EditUnsupported(_)));
}

#[test]
fn compact_limit_refused() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    {
        let mut b = FileBuilder::new();
        let ds = b.create_dataset("d").with_i32_data(&[1, 2, 3, 4]);
        for i in 0..8i64 {
            ds.set_attr(&format!("a{i}"), AttrValue::I64(i));
        }
        b.write(&p).unwrap();
    }

    let mut s = EditSession::open(&p).unwrap();
    s.set_dataset_attr("d", "overflow", AttrValue::I64(9)); // would be the 9th
    let err = s.commit().unwrap_err();
    assert!(matches!(err, Error::EditUnsupported(_)));
}

#[test]
fn set_attr_on_missing_dataset_refused() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build_contig(&p);

    let mut s = EditSession::open(&p).unwrap();
    s.set_dataset_attr("nope", "x", AttrValue::I64(1));
    let err = s.commit().unwrap_err();
    assert!(matches!(err, Error::EditUnsupported(_)));
}
