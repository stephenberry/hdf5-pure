//! Tests for in-place editing via `EditSession` (issue #32, Group C):
//! add, delete, and copy datasets and groups at any path.

use hdf5_pure::{AttrValue, DType, EditSession, File, FileBuilder, Object, ScaleOffset};

/// Write a starter file with one dataset, returning its path.
fn write_starter(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.create_dataset("original")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0]);
    b.write(path).unwrap();
}

#[test]
fn add_dataset_preserves_original_and_adds_new() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_one.h5");
    write_starter(&path);
    let size_before = std::fs::metadata(&path).unwrap().len();

    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_dataset("added").with_i32_data(&[10, 20, 30]);
        session.commit().unwrap();
    }

    // Only grew; existing bytes were not rewritten.
    let size_after = std::fs::metadata(&path).unwrap().len();
    assert!(size_after > size_before);

    let file = File::open(&path).unwrap();
    // Original dataset still intact.
    let orig = file.dataset("original").unwrap();
    assert_eq!(orig.dtype().unwrap(), DType::F64);
    assert_eq!(orig.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    // New dataset present and correct.
    let added = file.dataset("added").unwrap();
    assert_eq!(added.dtype().unwrap(), DType::I32);
    assert_eq!(added.read_i32().unwrap(), vec![10, 20, 30]);

    // Root group lists exactly the two datasets.
    let mut names = file.root().datasets().unwrap();
    names.sort();
    assert_eq!(names, vec!["added".to_string(), "original".to_string()]);

    std::fs::remove_file(&path).ok();
}

#[test]
fn add_multiple_datasets_in_one_commit() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_many.h5");
    write_starter(&path);

    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_dataset("a").with_f64_data(&[1.5, 2.5]);
        session.create_dataset("b").with_i32_data(&[7, 8, 9]);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("a").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5]
    );
    assert_eq!(
        file.dataset("b").unwrap().read_i32().unwrap(),
        vec![7, 8, 9]
    );
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn successive_commits_accumulate() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_successive.h5");
    write_starter(&path);

    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_dataset("first").with_i32_data(&[1]);
        session.commit().unwrap();
        session.create_dataset("second").with_i32_data(&[2]);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("first").unwrap().read_i32().unwrap(), vec![1]);
    assert_eq!(file.dataset("second").unwrap().read_i32().unwrap(), vec![2]);
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn add_dataset_with_multidim_shape() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_2d.h5");
    write_starter(&path);

    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("matrix")
            .with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .with_shape(&[2, 3]);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let m = file.dataset("matrix").unwrap();
    assert_eq!(m.shape().unwrap(), vec![2, 3]);
    assert_eq!(m.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    std::fs::remove_file(&path).ok();
}

#[test]
fn commit_without_staged_datasets_is_noop() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_noop.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session.commit().unwrap();
    }

    let after = std::fs::read(&path).unwrap();
    assert_eq!(before, after, "empty commit must not modify the file");
    std::fs::remove_file(&path).ok();
}

#[test]
fn create_group_at_root() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_group.h5");
    write_starter(&path);

    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_group("results");
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.root().groups().unwrap(), vec!["results".to_string()]);
    // The new group is empty and openable.
    assert!(
        file.group("results")
            .unwrap()
            .datasets()
            .unwrap()
            .is_empty()
    );
    // Original dataset intact.
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn add_dataset_into_new_nested_group() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_nested.h5");
    write_starter(&path);

    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_group("measurements");
        session.create_group("measurements/run1");
        session
            .create_dataset("measurements/run1/signal")
            .with_f64_data(&[10.0, 11.0, 12.0]);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("measurements/run1/signal").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![10.0, 11.0, 12.0]);
    // Ancestors and the original survive.
    assert_eq!(
        file.group("measurements").unwrap().groups().unwrap(),
        vec!["run1".to_string()]
    );
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn add_into_existing_group_across_commits() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_existing_group.h5");
    write_starter(&path);

    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_group("g");
        session.create_dataset("g/a").with_i32_data(&[1, 2]);
        session.commit().unwrap();
        // Second commit adds into the now-existing group g.
        session.create_dataset("g/b").with_i32_data(&[3, 4]);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("g/a").unwrap().read_i32().unwrap(), vec![1, 2]);
    assert_eq!(file.dataset("g/b").unwrap().read_i32().unwrap(), vec![3, 4]);
    let mut names = file.group("g").unwrap().datasets().unwrap();
    names.sort();
    assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
    std::fs::remove_file(&path).ok();
}

#[test]
fn add_into_two_sibling_groups_one_commit() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_siblings.h5");
    write_starter(&path);

    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_group("x");
        session.create_group("y");
        session.create_dataset("x/d").with_i32_data(&[1]);
        session.create_dataset("y/d").with_i32_data(&[2]);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("x/d").unwrap().read_i32().unwrap(), vec![1]);
    assert_eq!(file.dataset("y/d").unwrap().read_i32().unwrap(), vec![2]);
    std::fs::remove_file(&path).ok();
}

#[test]
fn dataset_into_missing_group_is_rejected() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_missing_group.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_dataset("nope/d").with_i32_data(&[1]);
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("does not exist"), "got: {err}");
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(&path).ok();
}

#[test]
fn duplicate_name_is_rejected_without_writing() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_dup.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    // Collide with the existing "original" dataset.
    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_dataset("original").with_i32_data(&[1, 2]);
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("already exists"), "got: {err}");
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);

    // Collide between two datasets staged in the same commit.
    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_dataset("dup").with_i32_data(&[1]);
        session.create_dataset("dup").with_i32_data(&[2]);
        assert!(session.commit().is_err());
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);

    std::fs::remove_file(&path).ok();
}

#[test]
fn delete_dataset_from_root() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_del_root.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.create_dataset("remove").with_i32_data(&[9, 9]);
    b.write(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session.delete("remove");
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert!(file.dataset("remove").is_err());
    std::fs::remove_file(&path).ok();
}

#[test]
fn delete_nested_group_subtree() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_del_nested.h5");
    write_starter(&path);
    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_group("grp");
        session.create_dataset("grp/inner").with_i32_data(&[5, 6]);
        session.create_dataset("sibling").with_i32_data(&[7]);
        session.commit().unwrap();
    }

    // Delete the whole group "grp" (its subtree becomes unreachable).
    {
        let mut session = EditSession::open(&path).unwrap();
        session.delete("grp");
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert!(file.group("grp").is_err());
    assert!(file.dataset("grp/inner").is_err());
    // Siblings and original survive.
    assert_eq!(
        file.dataset("sibling").unwrap().read_i32().unwrap(),
        vec![7]
    );
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    let mut roots = file.root().datasets().unwrap();
    roots.sort();
    assert_eq!(roots, vec!["original".to_string(), "sibling".to_string()]);
    assert!(file.root().groups().unwrap().is_empty());
    std::fs::remove_file(&path).ok();
}

#[test]
fn delete_one_of_nested_then_keep_group() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_del_one.h5");
    write_starter(&path);
    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_group("g");
        session.create_dataset("g/a").with_i32_data(&[1]);
        session.create_dataset("g/b").with_i32_data(&[2]);
        session.commit().unwrap();
    }
    {
        let mut session = EditSession::open(&path).unwrap();
        session.delete("g/a"); // remove one member, keep the group and g/b
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert!(file.dataset("g/a").is_err());
    assert_eq!(file.dataset("g/b").unwrap().read_i32().unwrap(), vec![2]);
    assert_eq!(
        file.group("g").unwrap().datasets().unwrap(),
        vec!["b".to_string()]
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn add_and_delete_in_one_commit() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_del.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("old").with_i32_data(&[1]);
    b.write(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_dataset("new").with_i32_data(&[2]);
        session.delete("old");
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["new".to_string()]);
    assert_eq!(file.dataset("new").unwrap().read_i32().unwrap(), vec![2]);
    assert!(file.dataset("old").is_err());
    std::fs::remove_file(&path).ok();
}

#[test]
fn delete_missing_or_overlapping_is_rejected() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_del_reject.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    // Nothing to delete.
    {
        let mut session = EditSession::open(&path).unwrap();
        session.delete("ghost");
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("nothing to delete"), "got: {err}");
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);

    // Delete /g while adding under it in the same commit → overlap rejected.
    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_group("g");
        session.commit().unwrap();
    }
    let mid = std::fs::read(&path).unwrap();
    {
        let mut session = EditSession::open(&path).unwrap();
        session.delete("g");
        session.create_dataset("g/x").with_i32_data(&[1]);
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("overlaps"), "got: {err}");
    }
    assert_eq!(std::fs::read(&path).unwrap(), mid);
    std::fs::remove_file(&path).ok();
}

#[test]
fn copy_dataset_to_new_name() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_copy_ds.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("src").with_f64_data(&[1.5, 2.5, 3.5]);
    b.write(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session.copy("src", "dup");
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    // Original and copy both present and identical.
    assert_eq!(
        file.dataset("src").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5, 3.5]
    );
    assert_eq!(
        file.dataset("dup").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5, 3.5]
    );
    assert_eq!(file.dataset("dup").unwrap().dtype().unwrap(), DType::F64);
    std::fs::remove_file(&path).ok();
}

#[test]
fn copy_group_subtree() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_copy_grp.h5");
    write_starter(&path);
    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_group("template");
        session.create_group("template/inner");
        session.create_dataset("template/a").with_i32_data(&[1, 2]);
        session
            .create_dataset("template/inner/b")
            .with_f64_data(&[9.0]);
        session.commit().unwrap();
    }

    // Copy the whole subtree under a new name.
    {
        let mut session = EditSession::open(&path).unwrap();
        session.copy("template", "run1");
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    // Deep structure duplicated.
    assert_eq!(
        file.dataset("run1/a").unwrap().read_i32().unwrap(),
        vec![1, 2]
    );
    assert_eq!(
        file.dataset("run1/inner/b").unwrap().read_f64().unwrap(),
        vec![9.0]
    );
    assert_eq!(
        file.group("run1").unwrap().groups().unwrap(),
        vec!["inner".to_string()]
    );
    // Original subtree untouched.
    assert_eq!(
        file.dataset("template/a").unwrap().read_i32().unwrap(),
        vec![1, 2]
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn copy_into_subgroup() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_copy_into.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("payload").with_i32_data(&[7, 8, 9]);
    b.write(&path).unwrap();
    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_group("dest");
        session.copy("payload", "dest/payload_copy");
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("dest/payload_copy")
            .unwrap()
            .read_i32()
            .unwrap(),
        vec![7, 8, 9]
    );
    assert_eq!(
        file.dataset("payload").unwrap().read_i32().unwrap(),
        vec![7, 8, 9]
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn copy_rejects_missing_source_and_cycle() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_copy_reject.h5");
    write_starter(&path);
    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_group("g");
        session.commit().unwrap();
    }
    let before = std::fs::read(&path).unwrap();

    // Missing source.
    {
        let mut session = EditSession::open(&path).unwrap();
        session.copy("ghost", "x");
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("source does not exist"),
            "got: {err}"
        );
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);

    // Copy a group into its own subtree.
    {
        let mut session = EditSession::open(&path).unwrap();
        session.copy("g", "g/inside");
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("itself"), "got: {err}");
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(&path).ok();
}

#[test]
fn add_dataset_with_attributes() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_attrs.h5");
    write_starter(&path);
    {
        let mut session = EditSession::open(&path).unwrap();
        let ds = session.create_dataset("measured");
        ds.with_f64_data(&[1.0, 2.0]);
        ds.set_attr("count", AttrValue::I64(2));
        ds.set_attr("unit", AttrValue::String("m/s".into()));
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("measured").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0]);
    let attrs = ds.attrs().unwrap();
    assert_eq!(attrs.get("count"), Some(&AttrValue::I64(2)));
    assert_eq!(attrs.get("unit"), Some(&AttrValue::String("m/s".into())));
    std::fs::remove_file(&path).ok();
}

#[test]
fn create_group_with_attributes() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_group_attrs.h5");
    write_starter(&path);
    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_group("run");
        session.set_group_attr("run", "kind", AttrValue::AsciiString("trial".into()));
        session.set_group_attr("run", "count", AttrValue::I64(2));
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let attrs = file.group("run").unwrap().attrs().unwrap();
    assert_eq!(attrs.get("kind"), Some(&AttrValue::String("trial".into())));
    assert_eq!(attrs.get("count"), Some(&AttrValue::I64(2)));
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn edit_existing_group_attributes() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_existing_group_attrs.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    g.set_attr("status", AttrValue::String("old".into()));
    g.set_attr("drop", AttrValue::I64(1));
    g.create_dataset("data").with_i32_data(&[5, 6]);
    b.add_group(g.finish());
    b.write(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session.set_group_attr("grp", "status", AttrValue::String("new".into()));
        session.set_group_attr("grp", "added", AttrValue::F64(3.5));
        session.remove_group_attr("grp", "drop");
        session.set_group_attr("/", "root_tag", AttrValue::U64(9));
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let grp_attrs = file.group("grp").unwrap().attrs().unwrap();
    assert_eq!(
        grp_attrs.get("status"),
        Some(&AttrValue::String("new".into()))
    );
    assert_eq!(grp_attrs.get("added"), Some(&AttrValue::F64(3.5)));
    assert!(!grp_attrs.contains_key("drop"));
    assert_eq!(
        file.group("grp").unwrap().datasets().unwrap(),
        vec!["data".to_string()]
    );
    assert_eq!(
        file.root().attrs().unwrap().get("root_tag"),
        Some(&AttrValue::U64(9))
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn group_attribute_edit_uses_final_compact_count() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_group_attr_final_count.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    for i in 0..8 {
        g.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    b.add_group(g.finish());
    b.write(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session.set_group_attr("grp", "new", AttrValue::I64(99));
        session.remove_group_attr("grp", "a0");
        session.commit().unwrap();
    }

    let attrs = File::open(&path)
        .unwrap()
        .group("grp")
        .unwrap()
        .attrs()
        .unwrap();
    assert_eq!(attrs.len(), 8);
    assert!(!attrs.contains_key("a0"));
    assert_eq!(attrs.get("new"), Some(&AttrValue::I64(99)));
    std::fs::remove_file(&path).ok();
}

#[test]
fn remove_missing_group_attribute_is_rejected_without_writing() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_missing_group_attr.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    g.set_attr("present", AttrValue::I64(1));
    b.add_group(g.finish());
    b.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session.remove_group_attr("grp", "missing");
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("not found"), "got: {err}");
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(&path).ok();
}

#[test]
fn add_variable_length_root_attribute_via_edit_session() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_vlen_group_attr.h5");
    write_starter(&path);

    {
        let mut session = EditSession::open(&path).unwrap();
        session.set_group_attr(
            "/",
            "fields",
            AttrValue::VarLenAsciiArray(vec!["a".into(), "b".into()]),
        );
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(
        attrs.get("fields"),
        Some(&AttrValue::StringArray(vec!["a".into(), "b".into()]))
    );
    // The rest of the file is untouched.
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn add_variable_length_group_attribute_then_remove_then_reset_in_one_commit() {
    // A Set/Remove/Set sequence for the same name in one commit must leave
    // only the final value, whether or not the intermediate states are
    // variable-length — exercising `apply_group_attr_ops`'s pending-VL-attr
    // bookkeeping (a plain region edit alone cannot represent an unresolved
    // variable-length attribute).
    let path = std::env::temp_dir().join("hdf5_pure_edit_vlen_group_attr_sequence.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    g.set_attr(
        "fields",
        AttrValue::VarLenAsciiArray(vec!["old1".into(), "old2".into()]),
    );
    b.add_group(g.finish());
    b.write(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        // Replace the existing variable-length attribute with a fixed-size
        // one, then remove it, then set a fresh variable-length value.
        session.set_group_attr("grp", "fields", AttrValue::I64(1));
        session.remove_group_attr("grp", "fields");
        session.set_group_attr(
            "grp",
            "fields",
            AttrValue::VarLenAsciiArray(vec!["new1".into(), "new2".into(), "new3".into()]),
        );
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let attrs = file.root().group("grp").unwrap().attrs().unwrap();
    assert_eq!(
        attrs.get("fields"),
        Some(&AttrValue::StringArray(vec![
            "new1".into(),
            "new2".into(),
            "new3".into()
        ]))
    );
    std::fs::remove_file(&path).ok();
}

/// A `Set` with a variable-length value must correctly drop a *fixed-size*
/// on-disk attribute of the same name, not just an existing pending
/// variable-length one: `apply_group_attr_ops`'s `remove_attr_from_region`
/// call is otherwise only exercised by the plain `Remove` op, never by a
/// variable-length `Set` replacing a fixed-size value.
#[test]
fn set_variable_length_group_attribute_over_existing_fixed_attribute_in_one_commit() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_vlen_group_attr_over_fixed.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    g.set_attr("fields", AttrValue::I64(42));
    b.add_group(g.finish());
    b.write(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session.set_group_attr(
            "grp",
            "fields",
            AttrValue::VarLenAsciiArray(vec!["new1".into(), "new2".into()]),
        );
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let attrs = file.root().group("grp").unwrap().attrs().unwrap();
    // Exactly one "fields" attribute survives, holding the new value — not a
    // leftover fixed-size copy alongside a new variable-length one.
    assert_eq!(attrs.len(), 1);
    assert_eq!(
        attrs.get("fields"),
        Some(&AttrValue::StringArray(vec!["new1".into(), "new2".into()]))
    );
    std::fs::remove_file(&path).ok();
}

/// The compact-attribute budget check counts *pending* variable-length
/// attributes alongside attributes already resolved into the region
/// (`compact_attr_count(&out)? + pending_vl.len()`); exactly at the boundary
/// (6 existing fixed + 2 new variable-length = 8 = `MAX_COMPACT_ATTRS`) must
/// still succeed, with every value intact.
#[test]
fn add_variable_length_group_attributes_at_budget_boundary_in_one_commit() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_vlen_group_attr_at_budget.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    for i in 0..6i64 {
        g.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    b.add_group(g.finish());
    b.write(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        // Two elements each: a single-element `VarLenAsciiArray` collapses to
        // `AttrValue::String` on read (matching every other array `AttrValue`
        // variant's len-1 collapse), which would make the read-back
        // assertions below ambiguous with a fixed-size string attribute.
        session.set_group_attr(
            "grp",
            "b0",
            AttrValue::VarLenAsciiArray(vec!["x0".into(), "x1".into()]),
        );
        session.set_group_attr(
            "grp",
            "b1",
            AttrValue::VarLenAsciiArray(vec!["y0".into(), "y1".into()]),
        );
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let attrs = file.root().group("grp").unwrap().attrs().unwrap();
    assert_eq!(attrs.len(), 8);
    for i in 0..6i64 {
        assert_eq!(attrs.get(&format!("a{i}")), Some(&AttrValue::I64(i)));
    }
    assert_eq!(
        attrs.get("b0"),
        Some(&AttrValue::StringArray(vec!["x0".into(), "x1".into()]))
    );
    assert_eq!(
        attrs.get("b1"),
        Some(&AttrValue::StringArray(vec!["y0".into(), "y1".into()]))
    );
    std::fs::remove_file(&path).ok();
}

/// One variable-length attribute past the boundary above (6 existing fixed +
/// 3 new variable-length = 9) is refused; since the 6 existing attributes
/// alone are under the budget, this specifically exercises the
/// `+ pending_vl.len()` term of the check (a regression here would let an
/// over-budget commit through without ever touching `compact_attr_count`'s
/// own counting logic).
#[test]
fn add_variable_length_group_attributes_over_budget_is_rejected_without_writing() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_vlen_group_attr_over_budget.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    for i in 0..6i64 {
        g.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    b.add_group(g.finish());
    b.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        for i in 0..3 {
            session.set_group_attr(
                "grp",
                &format!("b{i}"),
                AttrValue::VarLenAsciiArray(vec!["x".into()]),
            );
        }
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("dense"), "got: {err}");
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(&path).ok();
}

#[test]
fn dense_group_attribute_storage_is_still_rejected_without_writing() {
    // Dense (fractal-heap) attribute storage stays out of scope regardless of
    // whether the edit is fixed-size or variable-length; this guards that the
    // variable-length `Set` path added for issue #105 still refuses it rather
    // than silently mishandling it.
    let path = std::env::temp_dir().join("hdf5_pure_edit_dense_group_attr.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    for i in 0..12 {
        g.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    b.add_group(g.finish());
    b.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session.set_group_attr(
            "grp",
            "fields",
            AttrValue::VarLenAsciiArray(vec!["a".into(), "b".into()]),
        );
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("dense"), "got: {err}");
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(&path).ok();
}

#[test]
fn deleting_group_with_attribute_edit_is_rejected_without_writing() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_delete_group_attr_overlap.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    g.set_attr("tag", AttrValue::I64(1));
    b.add_group(g.finish());
    b.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session.delete("grp");
        session.set_group_attr("grp", "tag", AttrValue::I64(2));
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("overlaps"), "got: {err}");
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(&path).ok();
}

#[test]
fn copy_preserves_dataset_attributes() {
    // Exercises the "verbatim message bytes" claim: a copied dataset's
    // attributes (separate header messages) must survive byte-for-byte.
    let path = std::env::temp_dir().join("hdf5_pure_edit_copy_attrs.h5");
    let mut b = FileBuilder::new();
    let ds = b.create_dataset("src");
    ds.with_i32_data(&[5, 6, 7]);
    ds.set_attr("label", AttrValue::String("alpha".into()));
    ds.set_attr("scale", AttrValue::F64(2.5));
    b.write(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session.copy("src", "dup");
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let src_attrs = file.dataset("src").unwrap().attrs().unwrap();
    let dup = file.dataset("dup").unwrap();
    assert_eq!(dup.read_i32().unwrap(), vec![5, 6, 7]);
    // The copy's attributes equal the source's.
    assert_eq!(dup.attrs().unwrap(), src_attrs);
    assert_eq!(
        dup.attrs().unwrap().get("label"),
        Some(&AttrValue::String("alpha".into()))
    );
    std::fs::remove_file(&path).ok();
}

/// An unfiltered 2-D chunked dataset is copied: the values round-trip and the
/// copy is still chunked (the index is rebuilt at the new location).
#[test]
fn copy_unfiltered_chunked_dataset() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_copy_chunked.h5");
    let data: Vec<i32> = (0..24).collect();
    {
        let mut b = FileBuilder::new();
        b.create_dataset("src")
            .with_i32_data(&data)
            .with_shape(&[4, 6])
            .with_chunks(&[2, 3]);
        b.write(&path).unwrap();
    }
    {
        let mut session = EditSession::open(&path).unwrap();
        session.copy("src", "dup");
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    // Source untouched.
    assert_eq!(file.dataset("src").unwrap().read_i32().unwrap(), data);
    let dup = file.dataset("dup").unwrap();
    assert_eq!(dup.shape().unwrap(), vec![4, 6]);
    assert_eq!(dup.read_i32().unwrap(), data);
    assert!(
        dup.chunk_cache_stats().index_loaded(),
        "copied dataset must still be chunked"
    );
    std::fs::remove_file(&path).ok();
}

/// A filtered (shuffle + deflate) chunked dataset is copied verbatim: the chunk
/// bytes and filter pipeline are preserved (no recompression), the values round-
/// trip, the filter survives (the file stays far smaller than the raw bytes), and
/// the dataset's attributes are carried over.
#[test]
fn copy_filtered_chunked_dataset_preserves_pipeline_and_attrs() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_copy_filtered_chunked.h5");
    let data: Vec<i32> = (0..4096).map(|i| i % 4).collect(); // highly compressible
    {
        let mut b = FileBuilder::new();
        let ds = b.create_dataset("src");
        ds.with_i32_data(&data)
            .with_shape(&[4096])
            .with_chunks(&[512])
            .with_shuffle()
            .with_deflate(6);
        ds.set_attr("units", AttrValue::String("counts".into()));
        b.write(&path).unwrap();
    }
    {
        let mut session = EditSession::open(&path).unwrap();
        session.copy("src", "dup");
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    let dup = file.dataset("dup").unwrap();
    assert_eq!(dup.read_i32().unwrap(), data);
    assert!(
        dup.chunk_cache_stats().index_loaded(),
        "copied dataset must still be chunked"
    );
    // The filter survived: the whole file is far smaller than the raw element
    // bytes of a single copy, let alone two.
    assert!(
        std::fs::metadata(&path).unwrap().len() < (4096 * 4) as u64,
        "deflate filter must survive the copy"
    );
    // Attributes were preserved (the header is kept verbatim except its layout).
    assert_eq!(
        dup.attrs().unwrap().get("units"),
        Some(&AttrValue::String("counts".into()))
    );
    std::fs::remove_file(&path).ok();
}

/// An extensible (unlimited-dimension) chunked dataset copied within the file
/// stays readable; the copy uses an Extensible-Array index (selected from the
/// source's unlimited maxshape).
#[test]
fn copy_extensible_chunked_dataset() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_copy_extensible.h5");
    let data: Vec<f64> = (0..80).map(|i| i as f64 * 0.25).collect();
    write_starter(&path);
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("src")
            .with_f64_data(&data)
            .with_shape(&[80])
            .with_chunks(&[16])
            .with_maxshape(&[u64::MAX]);
        session.commit().unwrap();
    }
    {
        let mut session = EditSession::open(&path).unwrap();
        session.copy("src", "dup");
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    let dup = file.dataset("dup").unwrap();
    assert_eq!(dup.read_f64().unwrap(), data);
    assert!(dup.chunk_cache_stats().index_loaded());
    std::fs::remove_file(&path).ok();
}

/// A single-chunk dataset is copied (the chunk address lives in the layout
/// message; the verbatim path re-emits a single-chunk layout).
#[test]
fn copy_single_chunk_dataset() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_copy_single_chunk.h5");
    let data: Vec<i32> = (0..16).collect();
    {
        let mut b = FileBuilder::new();
        b.create_dataset("src")
            .with_i32_data(&data)
            .with_shape(&[16])
            .with_chunks(&[16]); // one chunk covers the whole dataset
        b.write(&path).unwrap();
    }
    {
        let mut session = EditSession::open(&path).unwrap();
        session.copy("src", "dup");
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("dup").unwrap().read_i32().unwrap(), data);
    std::fs::remove_file(&path).ok();
}

#[test]
fn edit_preserves_multiple_root_datasets() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_multi_root.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d0").with_i32_data(&[0]);
    b.create_dataset("d1").with_i32_data(&[1]);
    b.create_dataset("d2").with_i32_data(&[2]);
    b.write(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_group("extra");
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let mut names = file.root().datasets().unwrap();
    names.sort();
    assert_eq!(names, vec!["d0", "d1", "d2"]);
    for (i, n) in ["d0", "d1", "d2"].iter().enumerate() {
        assert_eq!(file.dataset(n).unwrap().read_i32().unwrap(), vec![i as i32]);
    }
    assert_eq!(file.root().groups().unwrap(), vec!["extra".to_string()]);
    std::fs::remove_file(&path).ok();
}

#[test]
fn mixed_add_delete_copy_in_one_commit() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_mixed.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 1]);
    b.create_dataset("remove").with_i32_data(&[9]);
    b.create_dataset("source").with_f64_data(&[3.0, 3.0, 3.0]);
    b.write(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_dataset("fresh").with_i32_data(&[42]); // add
        session.delete("remove"); // delete
        session.copy("source", "source_copy"); // copy
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let mut names = file.root().datasets().unwrap();
    names.sort();
    assert_eq!(names, vec!["fresh", "keep", "source", "source_copy"]);
    assert!(file.dataset("remove").is_err());
    assert_eq!(file.dataset("fresh").unwrap().read_i32().unwrap(), vec![42]);
    assert_eq!(
        file.dataset("source_copy").unwrap().read_f64().unwrap(),
        vec![3.0, 3.0, 3.0]
    );
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 1]
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn copy_from_file_dataset() {
    // Cross-file H5Ocopy: copy a dataset out of a separate open file.
    let src_path = std::env::temp_dir().join("hdf5_pure_xcopy_src_ds.h5");
    let dst_path = std::env::temp_dir().join("hdf5_pure_xcopy_dst_ds.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("payload").with_f64_data(&[1.5, 2.5, 3.5]);
        b.write(&src_path).unwrap();
    }
    write_starter(&dst_path);
    let src_bytes_before = std::fs::read(&src_path).unwrap();

    {
        let source = File::open(&src_path).unwrap();
        let mut session = EditSession::open(&dst_path).unwrap();
        session.copy_from(&source, "payload", "imported").unwrap();
        session.commit().unwrap();
    }

    // The copy landed in the destination, byte-equal to the source data.
    let file = File::open(&dst_path).unwrap();
    assert_eq!(
        file.dataset("imported").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5, 3.5]
    );
    assert_eq!(
        file.dataset("imported").unwrap().dtype().unwrap(),
        DType::F64
    );
    // The destination's pre-existing dataset is untouched.
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    // The source file was not modified at all.
    assert_eq!(std::fs::read(&src_path).unwrap(), src_bytes_before);

    std::fs::remove_file(&src_path).ok();
    std::fs::remove_file(&dst_path).ok();
}

#[test]
fn copy_from_file_group_subtree() {
    // A whole group subtree copied across files keeps its deep structure.
    let src_path = std::env::temp_dir().join("hdf5_pure_xcopy_src_grp.h5");
    let dst_path = std::env::temp_dir().join("hdf5_pure_xcopy_dst_grp.h5");
    write_starter(&src_path);
    {
        // Build the nested source subtree (FileBuilder::create_dataset does not
        // split paths into groups, so create the hierarchy explicitly).
        let mut s = EditSession::open(&src_path).unwrap();
        s.create_group("template");
        s.create_group("template/inner");
        s.create_dataset("template/a").with_i32_data(&[1, 2]);
        s.create_dataset("template/inner/b").with_f64_data(&[9.0]);
        s.commit().unwrap();
    }
    write_starter(&dst_path);

    {
        let source = File::open(&src_path).unwrap();
        let mut session = EditSession::open(&dst_path).unwrap();
        session.copy_from(&source, "template", "run1").unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&dst_path).unwrap();
    assert_eq!(
        file.dataset("run1/a").unwrap().read_i32().unwrap(),
        vec![1, 2]
    );
    assert_eq!(
        file.dataset("run1/inner/b").unwrap().read_f64().unwrap(),
        vec![9.0]
    );
    assert_eq!(
        file.group("run1").unwrap().groups().unwrap(),
        vec!["inner".to_string()]
    );
    std::fs::remove_file(&src_path).ok();
    std::fs::remove_file(&dst_path).ok();
}

#[test]
fn copy_from_file_into_subgroup_created_same_session() {
    // The destination parent may be a group created earlier in this session.
    let src_path = std::env::temp_dir().join("hdf5_pure_xcopy_src_into.h5");
    let dst_path = std::env::temp_dir().join("hdf5_pure_xcopy_dst_into.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("payload").with_i32_data(&[7, 8, 9]);
        b.write(&src_path).unwrap();
    }
    write_starter(&dst_path);

    {
        let source = File::open(&src_path).unwrap();
        let mut session = EditSession::open(&dst_path).unwrap();
        session.create_group("dest");
        session
            .copy_from(&source, "payload", "dest/payload_copy")
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&dst_path).unwrap();
    assert_eq!(
        file.dataset("dest/payload_copy")
            .unwrap()
            .read_i32()
            .unwrap(),
        vec![7, 8, 9]
    );
    std::fs::remove_file(&src_path).ok();
    std::fs::remove_file(&dst_path).ok();
}

#[test]
fn copy_from_file_preserves_attributes() {
    // Fixed-size attributes survive a cross-file copy byte-for-byte.
    let src_path = std::env::temp_dir().join("hdf5_pure_xcopy_src_attrs.h5");
    let dst_path = std::env::temp_dir().join("hdf5_pure_xcopy_dst_attrs.h5");
    {
        let mut b = FileBuilder::new();
        let ds = b.create_dataset("src");
        ds.with_i32_data(&[5, 6, 7]);
        ds.set_attr("label", AttrValue::String("alpha".into()));
        ds.set_attr("scale", AttrValue::F64(2.5));
        b.write(&src_path).unwrap();
    }
    write_starter(&dst_path);

    let src_attrs = {
        let source = File::open(&src_path).unwrap();
        let attrs = source.dataset("src").unwrap().attrs().unwrap();
        let mut session = EditSession::open(&dst_path).unwrap();
        session.copy_from(&source, "src", "dup").unwrap();
        session.commit().unwrap();
        attrs
    };

    let file = File::open(&dst_path).unwrap();
    let dup = file.dataset("dup").unwrap();
    assert_eq!(dup.read_i32().unwrap(), vec![5, 6, 7]);
    assert_eq!(dup.attrs().unwrap(), src_attrs);
    std::fs::remove_file(&src_path).ok();
    std::fs::remove_file(&dst_path).ok();
}

#[test]
fn copy_from_file_rejects_variable_length() {
    // A variable-length attribute stores global-heap references into the source
    // file; a verbatim cross-file copy cannot translate them, so it is refused.
    let src_path = std::env::temp_dir().join("hdf5_pure_xcopy_src_vlen.h5");
    let dst_path = std::env::temp_dir().join("hdf5_pure_xcopy_dst_vlen.h5");
    {
        let mut b = FileBuilder::new();
        let ds = b.create_dataset("src");
        ds.with_i32_data(&[1, 2, 3]);
        ds.set_attr(
            "tags",
            AttrValue::VarLenAsciiArray(vec!["one".into(), "two".into()]),
        );
        b.write(&src_path).unwrap();
    }
    write_starter(&dst_path);
    let dst_before = std::fs::read(&dst_path).unwrap();

    {
        let source = File::open(&src_path).unwrap();
        let mut session = EditSession::open(&dst_path).unwrap();
        let err = session.copy_from(&source, "src", "dup").unwrap_err();
        assert!(
            err.to_string().contains("variable-length or reference"),
            "got: {err}"
        );
        // Nothing was staged successfully, so a commit is a no-op.
        session.commit().unwrap();
    }

    // The destination is byte-unchanged; the same-file `copy` would have allowed
    // this (shared heap), but the cross-file path refuses it up front.
    assert_eq!(std::fs::read(&dst_path).unwrap(), dst_before);
    std::fs::remove_file(&src_path).ok();
    std::fs::remove_file(&dst_path).ok();
}

#[test]
fn copy_from_file_rejects_reference_dataset() {
    // An object-reference dataset stores absolute source-file object addresses; a
    // verbatim cross-file copy cannot translate them. This exercises the
    // datatype-message refusal branch (the variable-length test above covers the
    // attribute branch).
    let src_path = std::env::temp_dir().join("hdf5_pure_xcopy_src_ref.h5");
    let dst_path = std::env::temp_dir().join("hdf5_pure_xcopy_dst_ref.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("target").with_i32_data(&[1, 2, 3]);
        b.create_dataset("refs").with_path_references(&["target"]);
        b.write(&src_path).unwrap();
    }
    write_starter(&dst_path);
    let dst_before = std::fs::read(&dst_path).unwrap();

    {
        let source = File::open(&src_path).unwrap();
        let mut session = EditSession::open(&dst_path).unwrap();
        let err = session.copy_from(&source, "refs", "dup").unwrap_err();
        assert!(
            err.to_string().contains("variable-length or reference"),
            "got: {err}"
        );
    }
    assert_eq!(std::fs::read(&dst_path).unwrap(), dst_before);
    std::fs::remove_file(&src_path).ok();
    std::fs::remove_file(&dst_path).ok();
}

#[test]
fn copy_from_file_rejects_missing_source() {
    let src_path = std::env::temp_dir().join("hdf5_pure_xcopy_src_missing.h5");
    let dst_path = std::env::temp_dir().join("hdf5_pure_xcopy_dst_missing.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("present").with_i32_data(&[1]);
        b.write(&src_path).unwrap();
    }
    write_starter(&dst_path);

    let source = File::open(&src_path).unwrap();
    let mut session = EditSession::open(&dst_path).unwrap();
    let err = session.copy_from(&source, "ghost", "x").unwrap_err();
    assert!(err.to_string().contains("does not exist"), "got: {err}");

    std::fs::remove_file(&src_path).ok();
    std::fs::remove_file(&dst_path).ok();
}

#[test]
fn copy_from_file_rejects_destination_collision() {
    // A destination name already present in the parent group is refused at commit,
    // leaving the file untouched.
    let src_path = std::env::temp_dir().join("hdf5_pure_xcopy_src_collide.h5");
    let dst_path = std::env::temp_dir().join("hdf5_pure_xcopy_dst_collide.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("payload").with_i32_data(&[1]);
        b.write(&src_path).unwrap();
    }
    write_starter(&dst_path); // contains "original"
    let dst_before = std::fs::read(&dst_path).unwrap();

    {
        let source = File::open(&src_path).unwrap();
        let mut session = EditSession::open(&dst_path).unwrap();
        session.copy_from(&source, "payload", "original").unwrap();
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("already exists"), "got: {err}");
    }
    assert_eq!(std::fs::read(&dst_path).unwrap(), dst_before);

    std::fs::remove_file(&src_path).ok();
    std::fs::remove_file(&dst_path).ok();
}

#[test]
fn copy_from_file_rejects_streaming_source() {
    // The source must be buffered so its bytes are addressable; a streaming reader
    // is refused with a clear message.
    let src_path = std::env::temp_dir().join("hdf5_pure_xcopy_src_stream.h5");
    let dst_path = std::env::temp_dir().join("hdf5_pure_xcopy_dst_stream.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("payload").with_i32_data(&[1, 2, 3]);
        b.write(&src_path).unwrap();
    }
    write_starter(&dst_path);

    let source = File::open_streaming(&src_path).unwrap();
    let mut session = EditSession::open(&dst_path).unwrap();
    let err = session.copy_from(&source, "payload", "dup").unwrap_err();
    assert!(err.to_string().contains("buffered source"), "got: {err}");

    std::fs::remove_file(&src_path).ok();
    std::fs::remove_file(&dst_path).ok();
}

#[test]
fn copy_same_file_still_allows_variable_length_attribute() {
    // Regression guard: the foreign-address refusal is cross-file only. An in-file
    // `copy` of a dataset carrying a variable-length attribute still works (the
    // copy shares the source file's global heap).
    let path = std::env::temp_dir().join("hdf5_pure_xcopy_infile_vlen.h5");
    {
        let mut b = FileBuilder::new();
        let ds = b.create_dataset("src");
        ds.with_i32_data(&[1, 2, 3]);
        ds.set_attr(
            "tags",
            AttrValue::VarLenAsciiArray(vec!["one".into(), "two".into()]),
        );
        b.write(&path).unwrap();
    }

    {
        let mut session = EditSession::open(&path).unwrap();
        session.copy("src", "dup");
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let src_attrs = file.dataset("src").unwrap().attrs().unwrap();
    assert_eq!(file.dataset("dup").unwrap().attrs().unwrap(), src_attrs);
    assert_eq!(
        file.dataset("dup").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn superblock_eof_matches_file_size_after_edit() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_eof.h5");
    write_starter(&path);
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("more")
            .with_f64_data(&[1.0, 2.0, 3.0]);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let on_disk = std::fs::metadata(&path).unwrap().len();
    // The edit updates the superblock's logical end-of-file to the new size.
    assert_eq!(file.file_size(), on_disk);
    assert_eq!(file.superblock().eof_address, on_disk);
    std::fs::remove_file(&path).ok();
}

/// A chunked (but unfiltered) dataset can be added in place and read back, and
/// the original dataset is left intact.
#[test]
fn add_chunked_dataset() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_chunked.h5");
    write_starter(&path);

    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("chunky")
            .with_f64_data(&data)
            .with_chunks(&[25]);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let chunky = file.dataset("chunky").unwrap();
    assert_eq!(chunky.shape().unwrap(), vec![100]);
    assert_eq!(chunky.read_f64().unwrap(), data);
    // Original dataset untouched.
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    // The superblock's end-of-file matches the physical size after appending the
    // chunk data, index, and header.
    assert_eq!(file.file_size(), std::fs::metadata(&path).unwrap().len());
    std::fs::remove_file(&path).ok();
}

/// Deflate, shuffle+deflate, and fletcher32 filtered datasets each round-trip
/// through the in-place editor and the reader.
#[test]
fn add_filtered_datasets() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_filtered.h5");
    write_starter(&path);

    let data: Vec<f64> = (0..200).map(|i| i as f64).collect();
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("deflated")
            .with_f64_data(&data)
            .with_chunks(&[50])
            .with_deflate(6);
        session
            .create_dataset("shuffled")
            .with_f64_data(&data)
            .with_chunks(&[50])
            .with_shuffle()
            .with_deflate(4);
        session
            .create_dataset("checked")
            .with_f64_data(&data)
            .with_chunks(&[64])
            .with_fletcher32();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    for name in ["deflated", "shuffled", "checked"] {
        assert_eq!(
            file.dataset(name).unwrap().read_f64().unwrap(),
            data,
            "dataset {name} did not round-trip"
        );
    }
    assert_eq!(file.file_size(), std::fs::metadata(&path).unwrap().len());
    std::fs::remove_file(&path).ok();
}

/// A lossless integer scale-offset dataset round-trips.
#[test]
fn add_scale_offset_dataset() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_scaleoffset.h5");
    write_starter(&path);

    let data: Vec<i32> = (0..120).map(|i| 1000 + (i % 7)).collect();
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("counts")
            .with_i32_data(&data)
            .with_chunks(&[40])
            .with_scale_offset(ScaleOffset::Integer(0));
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("counts").unwrap().read_i32().unwrap(), data);
    std::fs::remove_file(&path).ok();
}

/// A 2-D chunked dataset whose chunks don't evenly divide the shape round-trips
/// (exercises edge chunks and the fixed-array index used for >1 chunk).
#[test]
fn add_2d_chunked_dataset() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_2d_chunked.h5");
    write_starter(&path);

    let data: Vec<i32> = (0..(7 * 5)).collect();
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("grid")
            .with_i32_data(&data)
            .with_shape(&[7, 5])
            .with_chunks(&[3, 2]);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let grid = file.dataset("grid").unwrap();
    assert_eq!(grid.shape().unwrap(), vec![7, 5]);
    assert_eq!(grid.read_i32().unwrap(), data);
    std::fs::remove_file(&path).ok();
}

/// An extensible (unlimited-dimension) dataset can be added in place; its data
/// reads back and the file remains valid. The unlimited dimension selects the
/// Extensible-Array chunk index.
#[test]
fn add_extensible_dataset() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_extensible.h5");
    write_starter(&path);

    let data: Vec<i32> = (0..64).collect();
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("stream")
            .with_i32_data(&data)
            .with_shape(&[64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[16]);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let stream = file.dataset("stream").unwrap();
    assert_eq!(stream.shape().unwrap(), vec![64]);
    assert_eq!(stream.read_i32().unwrap(), data);
    assert_eq!(file.file_size(), std::fs::metadata(&path).unwrap().len());
    std::fs::remove_file(&path).ok();
}

/// One commit can mix a contiguous dataset and a chunked/compressed dataset
/// into a nested group, alongside the original.
#[test]
fn add_mixed_contiguous_and_chunked_in_group() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_mixed.h5");
    write_starter(&path);

    let wave: Vec<f64> = (0..512).map(|i| (i as f64 * 0.1).cos()).collect();
    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_group("run");
        session
            .create_dataset("run/scalarish")
            .with_i32_data(&[1, 2, 3]);
        session
            .create_dataset("run/wave")
            .with_f64_data(&wave)
            .with_chunks(&[128])
            .with_shuffle()
            .with_deflate(6);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("run/scalarish").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(file.dataset("run/wave").unwrap().read_f64().unwrap(), wave);
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(file.file_size(), std::fs::metadata(&path).unwrap().len());
    std::fs::remove_file(&path).ok();
}

/// A chunked dataset whose datatype is `f64` still reports the right dtype after
/// an in-place add, confirming the header is a faithful chunked dataset header.
#[test]
fn added_chunked_dataset_reports_dtype() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_chunked_dtype.h5");
    write_starter(&path);
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("c")
            .with_f64_data(&(0..50).map(f64::from).collect::<Vec<_>>())
            .with_chunks(&[10])
            .with_deflate(3);
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("c").unwrap().dtype().unwrap(), DType::F64);
    std::fs::remove_file(&path).ok();
}

/// A ZFP fixed-rate compressed dataset can be added in place and reads back
/// within ZFP's lossy tolerance (gated on the `zfp` feature).
#[test]
#[cfg(feature = "zfp")]
fn add_zfp_dataset() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_zfp.h5");
    write_starter(&path);

    let data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.05).sin()).collect();
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("zfp")
            .with_f64_data(&data)
            .with_chunks(&[64])
            .with_zfp(32.0);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let back = file.dataset("zfp").unwrap().read_f64().unwrap();
    assert_eq!(back.len(), data.len());
    let max_err = data
        .iter()
        .zip(back.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0f64, f64::max);
    assert!(max_err < 1e-6, "ZFP max_err {max_err} > 1e-6");
    std::fs::remove_file(&path).ok();
}

/// Malformed chunked-dataset requests are refused before any byte is written,
/// rather than panicking or producing a silently-corrupt dataset: an empty
/// (zero-element) shape, chunk dims whose rank disagrees with the shape, a zero
/// chunk dimension, and a maxshape whose rank disagrees with the shape.
#[test]
fn malformed_chunked_requests_are_rejected_without_writing() {
    // A no-capture configurator per malformed case; `fn` pointers keep the case
    // table a simple type.
    type Configure = fn(&mut hdf5_pure::DatasetBuilder);
    let bad: &[(&str, Configure)] = &[
        ("empty shape", |b| {
            b.with_f64_data(&[]).with_shape(&[0]).with_chunks(&[4]);
        }),
        ("chunk rank mismatch", |b| {
            b.with_i32_data(&[1, 2, 3, 4, 5, 6])
                .with_shape(&[2, 3])
                .with_chunks(&[2]);
        }),
        ("zero chunk dim", |b| {
            b.with_i32_data(&[1, 2, 3, 4])
                .with_shape(&[4])
                .with_chunks(&[0]);
        }),
        ("maxshape rank mismatch", |b| {
            b.with_i32_data(&[1, 2, 3, 4])
                .with_shape(&[4])
                .with_maxshape(&[u64::MAX, u64::MAX])
                .with_chunks(&[2]);
        }),
        ("scalar with chunks", |b| {
            b.with_f64_data(&[1.0]).with_shape(&[]).with_chunks(&[1]);
        }),
        ("maxshape below shape", |b| {
            b.with_i32_data(&[1, 2, 3, 4])
                .with_shape(&[4])
                .with_maxshape(&[2]);
        }),
    ];

    for (label, configure) in bad {
        let path = std::env::temp_dir().join(format!(
            "hdf5_pure_edit_reject_{}.h5",
            label.replace(' ', "_")
        ));
        write_starter(&path);
        let before = std::fs::read(&path).unwrap();
        {
            let mut session = EditSession::open(&path).unwrap();
            configure(session.create_dataset("bad"));
            let err = session.commit().unwrap_err();
            assert!(
                err.to_string().contains("in-place edit"),
                "[{label}] expected an EditUnsupported refusal, got: {err}"
            );
        }
        // The guard runs before any write, so the file is untouched.
        assert_eq!(
            std::fs::read(&path).unwrap(),
            before,
            "[{label}] file modified"
        );
        std::fs::remove_file(&path).ok();
    }
}

// ---- write_dataset: in-place value overwrite (issue #79) ----

#[test]
fn write_dataset_same_size_overwrites_in_place() {
    let path = std::env::temp_dir().join("hdf5_pure_write_same_size.h5");
    write_starter(&path); // "original" = [1.0, 2.0, 3.0, 4.0] (contiguous f64)
    let size_before = std::fs::metadata(&path).unwrap().len();

    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .write_dataset("original")
            .with_f64_data(&[9.0, 8.0, 7.0, 6.0]);
        session.commit().unwrap();
    }

    // Same-length overwrite reuses the existing block: the file does not grow.
    let size_after = std::fs::metadata(&path).unwrap().len();
    assert_eq!(
        size_after, size_before,
        "same-size write should not grow file"
    );

    let file = File::open(&path).unwrap();
    let ds = file.dataset("original").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::F64);
    assert_eq!(ds.read_f64().unwrap(), vec![9.0, 8.0, 7.0, 6.0]);
    std::fs::remove_file(&path).ok();
}

#[test]
fn write_dataset_resize_keeping_shape_is_a_reshape_and_refused() {
    // For a fixed-size datatype the byte length is shape * element size, so a
    // different-length replacement necessarily changes the shape — which is a
    // reshape, not a value overwrite, and is refused. (The genuine relocation
    // path — overwriting a never-written, undefined-address dataset — is exercised
    // in the crosscheck against the C library, which can create one.)
    let path = std::env::temp_dir().join("hdf5_pure_write_resize_refused.h5");
    write_starter(&path); // "original" = 4 f64
    let before = std::fs::read(&path).unwrap();
    {
        let mut session = EditSession::open(&path).unwrap();
        session.write_dataset("original").with_f64_data(&[42.0]);
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("shape does not match"),
            "expected reshape refusal, got: {err}"
        );
    }
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "file modified on refusal"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn write_dataset_in_a_nested_group() {
    let path = std::env::temp_dir().join("hdf5_pure_write_nested.h5");
    {
        let mut b = FileBuilder::new();
        let mut g = b.create_group("grp");
        g.create_dataset("inner").with_i32_data(&[1, 2, 3]);
        b.add_group(g.finish());
        b.write(&path).unwrap();
    }
    {
        let mut session = EditSession::open(&path).unwrap();
        // Same size (in place) for the nested dataset.
        session
            .write_dataset("grp/inner")
            .with_i32_data(&[10, 20, 30]);
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("grp/inner").unwrap().read_i32().unwrap(),
        vec![10, 20, 30]
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn write_dataset_rejects_datatype_mismatch() {
    let path = std::env::temp_dir().join("hdf5_pure_write_type_mismatch.h5");
    write_starter(&path); // f64
    let before = std::fs::read(&path).unwrap();
    {
        let mut session = EditSession::open(&path).unwrap();
        // i32 data for an f64 dataset: a retype, refused.
        session
            .write_dataset("original")
            .with_i32_data(&[1, 2, 3, 4]);
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("datatype does not match"),
            "expected datatype-mismatch refusal, got: {err}"
        );
    }
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "file modified on refusal"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn write_dataset_rejects_shape_mismatch() {
    let path = std::env::temp_dir().join("hdf5_pure_write_shape_mismatch.h5");
    {
        let mut b = FileBuilder::new();
        // A 2-D dataset, so a 1-D replacement of the same element count is a
        // reshape (different dataspace bytes), which is refused.
        b.create_dataset("m")
            .with_i32_data(&[1, 2, 3, 4, 5, 6])
            .with_shape(&[2, 3]);
        b.write(&path).unwrap();
    }
    let before = std::fs::read(&path).unwrap();
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .write_dataset("m")
            .with_i32_data(&[1, 2, 3, 4, 5, 6])
            .with_shape(&[6]);
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("shape does not match"),
            "expected shape-mismatch refusal, got: {err}"
        );
    }
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "file modified on refusal"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn write_dataset_rejects_missing_target() {
    let path = std::env::temp_dir().join("hdf5_pure_write_missing.h5");
    write_starter(&path);
    {
        let mut session = EditSession::open(&path).unwrap();
        session.write_dataset("nope").with_f64_data(&[1.0]);
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("nothing to overwrite"),
            "expected missing-target refusal, got: {err}"
        );
    }
    std::fs::remove_file(&path).ok();
}

/// An unfiltered chunked dataset is overwritten chunk-by-chunk straight in its
/// existing slots: the file does not grow (no header rewrite, no index change)
/// and the new values read back.
#[test]
fn write_dataset_overwrites_unfiltered_chunked_in_place() {
    let path = std::env::temp_dir().join("hdf5_pure_write_chunked.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("c")
            .with_i32_data(&[1, 2, 3, 4, 5, 6, 7, 8])
            .with_shape(&[8])
            .with_chunks(&[4]);
        b.write(&path).unwrap();
    }
    let size_before = std::fs::metadata(&path).unwrap().len();
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .write_dataset("c")
            .with_i32_data(&[8, 7, 6, 5, 4, 3, 2, 1])
            .with_shape(&[8]);
        session.commit().unwrap();
    }
    // An unfiltered chunked overwrite is a true in-place write: the chunk slots
    // are reused, so the file did not grow.
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        size_before,
        "unfiltered chunked overwrite should not grow the file"
    );
    let file = File::open(&path).unwrap();
    let c = file.dataset("c").unwrap();
    assert_eq!(c.shape().unwrap(), vec![8]);
    assert_eq!(c.read_i32().unwrap(), vec![8, 7, 6, 5, 4, 3, 2, 1]);
    assert!(
        c.chunk_cache_stats().index_loaded(),
        "dataset must still be chunked"
    );
    std::fs::remove_file(&path).ok();
}

/// A 2-D chunked dataset whose chunks do not evenly divide the shape (edge
/// chunks, Fixed-Array index) is overwritten in place.
#[test]
fn write_dataset_overwrites_2d_edge_chunked_in_place() {
    let path = std::env::temp_dir().join("hdf5_pure_write_2d_edge_chunked.h5");
    let orig: Vec<i32> = (0..35).collect();
    {
        let mut b = FileBuilder::new();
        b.create_dataset("g")
            .with_i32_data(&orig)
            .with_shape(&[7, 5])
            .with_chunks(&[3, 2]);
        b.write(&path).unwrap();
    }
    let size_before = std::fs::metadata(&path).unwrap().len();
    let updated: Vec<i32> = orig.iter().rev().copied().collect();
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .write_dataset("g")
            .with_i32_data(&updated)
            .with_shape(&[7, 5]);
        session.commit().unwrap();
    }
    assert_eq!(std::fs::metadata(&path).unwrap().len(), size_before);
    let file = File::open(&path).unwrap();
    let g = file.dataset("g").unwrap();
    assert_eq!(g.shape().unwrap(), vec![7, 5]);
    assert_eq!(g.read_i32().unwrap(), updated);
    std::fs::remove_file(&path).ok();
}

/// An extensible (unlimited-dimension, Extensible-Array index) chunked dataset is
/// overwritten in place.
#[test]
fn write_dataset_overwrites_extensible_chunked_in_place() {
    let path = std::env::temp_dir().join("hdf5_pure_write_extensible_chunked.h5");
    let orig: Vec<f64> = (0..60).map(|i| i as f64).collect();
    write_starter(&path);
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("ext")
            .with_f64_data(&orig)
            .with_shape(&[60])
            .with_chunks(&[16])
            .with_maxshape(&[u64::MAX]);
        session.commit().unwrap();
    }
    let updated: Vec<f64> = orig.iter().map(|v| v * 2.0).collect();
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .write_dataset("ext")
            .with_f64_data(&updated)
            .with_shape(&[60]);
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("ext").unwrap().read_f64().unwrap(), updated);
    std::fs::remove_file(&path).ok();
}

/// A size-preserving filter (Fletcher32 always appends a 4-byte checksum, so the
/// stored size is independent of the values) lets a filtered chunked dataset be
/// overwritten in place even when the values change.
#[test]
fn write_dataset_overwrites_fletcher32_chunked_in_place() {
    let path = std::env::temp_dir().join("hdf5_pure_write_fletcher_chunked.h5");
    let orig: Vec<f64> = (0..128).map(|i| i as f64).collect();
    write_starter(&path);
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("ck")
            .with_f64_data(&orig)
            .with_shape(&[128])
            .with_chunks(&[64])
            .with_fletcher32();
        session.commit().unwrap();
    }
    let size_before = std::fs::metadata(&path).unwrap().len();
    let updated: Vec<f64> = orig.iter().map(|v| v + 1000.0).collect();
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .write_dataset("ck")
            .with_f64_data(&updated)
            .with_shape(&[128]);
        session.commit().unwrap();
    }
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        size_before,
        "a Fletcher32 overwrite keeps each chunk's stored size, so it stays in place"
    );
    let file = File::open(&path).unwrap();
    let ck = file.dataset("ck").unwrap();
    assert_eq!(ck.read_f64().unwrap(), updated);
    assert!(ck.chunk_cache_stats().index_loaded());
    std::fs::remove_file(&path).ok();
}

/// Rewriting a deflate dataset with the *same* values reproduces identical
/// compressed bytes, so the overwrite fits the existing slots and stays in place.
#[test]
fn write_dataset_overwrites_deflate_chunked_equal_size_in_place() {
    let path = std::env::temp_dir().join("hdf5_pure_write_deflate_equal.h5");
    let data: Vec<f64> = (0..200).map(|i| i as f64).collect();
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_f64_data(&data)
            .with_shape(&[200])
            .with_chunks(&[50])
            .with_deflate(6);
        b.write(&path).unwrap();
    }
    let size_before = std::fs::metadata(&path).unwrap().len();
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .write_dataset("d")
            .with_f64_data(&data)
            .with_shape(&[200]);
        session.commit().unwrap();
    }
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        size_before,
        "re-encoding identical values is byte-identical, so the overwrite stays in place"
    );
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_f64().unwrap(), data);
    std::fs::remove_file(&path).ok();
}

/// A deflate dataset overwritten with values of different compressibility
/// re-encodes to a different size, so the dataset is rebuilt and relocated; the
/// new values still read back and the dataset stays chunked + compressed.
#[test]
fn write_dataset_overwrites_deflate_chunked_relocates_on_size_change() {
    let path = std::env::temp_dir().join("hdf5_pure_write_deflate_relocate.h5");
    // Highly compressible original, then incompressible-ish replacement.
    let orig: Vec<i32> = vec![0; 4096];
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&orig)
            .with_shape(&[4096])
            .with_chunks(&[512])
            .with_deflate(6);
        b.write(&path).unwrap();
    }
    let updated: Vec<i32> = (0..4096i32)
        .map(|i| i.wrapping_mul(2_654_435_761u32 as i32) ^ (i << 3))
        .collect();
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .write_dataset("d")
            .with_i32_data(&updated)
            .with_shape(&[4096]);
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    let d = file.dataset("d").unwrap();
    assert_eq!(d.read_i32().unwrap(), updated);
    assert!(
        d.chunk_cache_stats().index_loaded(),
        "relocated dataset must still be chunked"
    );
    std::fs::remove_file(&path).ok();
}

/// A relocating chunked overwrite returns the old chunk storage to the session's
/// free list (the same path the delete reclaim uses), and a subsequent addition
/// in the same session draws from it. This exercises the relocate -> free ->
/// reuse interplay: the file must stay valid and both datasets read back exactly
/// (a double-free or stale span would corrupt one of them).
#[test]
fn write_dataset_chunked_relocate_then_reuse_stays_valid() {
    let path = std::env::temp_dir().join("hdf5_pure_write_chunked_reclaim.h5");
    {
        let mut b = FileBuilder::new();
        // Highly compressible start => tiny chunk slots.
        b.create_dataset("d")
            .with_i32_data(&vec![0i32; 4096])
            .with_shape(&[4096])
            .with_chunks(&[512])
            .with_deflate(6);
        b.write(&path).unwrap();
    }
    // Incompressible new values grow the chunks past their tiny slots, forcing a
    // relocate that frees the old chunk storage into the session free list.
    let updated: Vec<i32> = (0..4096i32)
        .map(|i| i.wrapping_mul(2_654_435_761u32 as i32) ^ (i << 3))
        .collect();
    let filler: Vec<f64> = (0..64).map(|i| i as f64).collect();
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .write_dataset("d")
            .with_i32_data(&updated)
            .with_shape(&[4096]);
        session.commit().unwrap();

        // A later addition in the same session draws from the freed regions.
        session.create_dataset("filler").with_f64_data(&filler);
        session.commit().unwrap();
    } // drop the editor (release its file lock) before reading back

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), updated);
    assert_eq!(file.dataset("filler").unwrap().read_f64().unwrap(), filler);
    std::fs::remove_file(&path).ok();
}

/// A filtered (deflate, Fixed-Array index) overwrite whose re-encoded chunks are
/// *smaller* than their slots is applied in place: each shrunk chunk is written
/// into its existing slot and the chunk index is rebuilt in place to record the
/// new sizes, so the file does not grow and the new values read back (which would
/// be impossible if the index still recorded the old, larger sizes).
#[test]
fn write_dataset_overwrites_filtered_chunked_fits_with_slack_in_place() {
    let path = std::env::temp_dir().join("hdf5_pure_write_fits_slack_fa.h5");
    // Incompressible start => large chunk slots (Fixed Array: 4 finite chunks).
    let orig: Vec<i32> = (0..2048i32)
        .map(|i| i.wrapping_mul(2_654_435_761u32 as i32) ^ (i << 3))
        .collect();
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&orig)
            .with_shape(&[2048])
            .with_chunks(&[512])
            .with_deflate(6);
        b.write(&path).unwrap();
    }
    let size_before = std::fs::metadata(&path).unwrap().len();
    // Highly compressible replacement => much smaller chunks that fit with slack.
    let updated: Vec<i32> = vec![7; 2048];
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .write_dataset("d")
            .with_i32_data(&updated)
            .with_shape(&[2048]);
        session.commit().unwrap();
    }
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        size_before,
        "a fits-with-slack overwrite reuses the chunk slots and rebuilds the index \
         in place, so the file must not grow"
    );
    let file = File::open(&path).unwrap();
    let d = file.dataset("d").unwrap();
    assert_eq!(d.read_i32().unwrap(), updated);
    assert!(d.chunk_cache_stats().index_loaded(), "still chunked");
    std::fs::remove_file(&path).ok();
}

/// The fits-with-slack in-place path also covers an extensible (unlimited,
/// Extensible-Array index) dataset: the shrunk chunks reuse their slots and the
/// EA index is rebuilt in place, so the file does not grow and the values read
/// back.
#[test]
fn write_dataset_overwrites_filtered_extensible_fits_with_slack() {
    let path = std::env::temp_dir().join("hdf5_pure_write_fits_slack_ea.h5");
    let orig: Vec<i32> = (0..2048i32)
        .map(|i| i.wrapping_mul(2_654_435_761u32 as i32) ^ (i << 3))
        .collect();
    write_starter(&path);
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("d")
            .with_i32_data(&orig)
            .with_shape(&[2048])
            .with_chunks(&[512])
            .with_maxshape(&[u64::MAX]) // unlimited => Extensible Array index
            .with_deflate(6);
        session.commit().unwrap();
    }
    let size_before = std::fs::metadata(&path).unwrap().len();
    let updated: Vec<i32> = vec![3; 2048];
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .write_dataset("d")
            .with_i32_data(&updated)
            .with_shape(&[2048]);
        session.commit().unwrap();
    }
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        size_before,
        "an extensible-array fits-with-slack overwrite also rebuilds the index in \
         place, so the file must not grow"
    );
    let file = File::open(&path).unwrap();
    let d = file.dataset("d").unwrap();
    assert_eq!(d.read_i32().unwrap(), updated);
    assert!(d.chunk_cache_stats().index_loaded(), "still chunked");
    std::fs::remove_file(&path).ok();
}

#[test]
fn write_dataset_rejects_filtered_request() {
    // A builder that itself requests chunking/filtering is refused as "not a
    // value overwrite" before the on-disk dataset is even consulted.
    let path = std::env::temp_dir().join("hdf5_pure_write_filtered_request.h5");
    write_starter(&path);
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .write_dataset("original")
            .with_f64_data(&[1.0, 2.0, 3.0, 4.0])
            .with_shape(&[4])
            .with_chunks(&[2]);
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("overwrites values only"),
            "expected value-only refusal, got: {err}"
        );
    }
    std::fs::remove_file(&path).ok();
}

#[test]
fn write_dataset_rejects_staged_attributes() {
    // Attributes set on the write_dataset builder cannot be applied by a value
    // overwrite, so they must be refused rather than silently dropped.
    let path = std::env::temp_dir().join("hdf5_pure_write_attr_refused.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .write_dataset("original")
            .with_f64_data(&[5.0, 6.0, 7.0, 8.0]) // same size, valid overwrite
            .set_attr("units", AttrValue::String("m/s".into()));
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("cannot set attributes"),
            "expected attribute refusal, got: {err}"
        );
    }
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "file modified on refusal"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn write_dataset_alongside_other_edits() {
    // A value overwrite coexists with an addition and a delete in one commit.
    let path = std::env::temp_dir().join("hdf5_pure_write_mixed.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("keep").with_f64_data(&[1.0, 2.0]);
        b.create_dataset("doomed").with_i32_data(&[9]);
        b.write(&path).unwrap();
    }
    {
        let mut session = EditSession::open(&path).unwrap();
        session.write_dataset("keep").with_f64_data(&[5.0, 6.0]); // same size
        session.create_dataset("added").with_i32_data(&[3, 4]);
        session.delete("doomed");
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("keep").unwrap().read_f64().unwrap(),
        vec![5.0, 6.0]
    );
    assert_eq!(
        file.dataset("added").unwrap().read_i32().unwrap(),
        vec![3, 4]
    );
    assert!(file.dataset("doomed").is_err());
    std::fs::remove_file(&path).ok();
}

#[test]
fn write_dataset_with_no_other_edits_takes_inplace_fast_path() {
    // A lone same-size overwrite must not rewrite headers or flip the root: the
    // only on-disk bytes that change are the data block itself.
    let path = std::env::temp_dir().join("hdf5_pure_write_fastpath.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .write_dataset("original")
            .with_f64_data(&[1.0, 2.0, 3.0, 4.0]); // identical bytes
        session.commit().unwrap();
    }
    // Identical data written back in place leaves the file byte-for-byte the same.
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "in-place rewrite of identical data changed the file"
    );
    std::fs::remove_file(&path).ok();
}

/// A zero-element (empty) contiguous dataset — the on-disk equivalent of the
/// whole-file writer's `HADDR_UNDEF` data address for "no storage allocated"
/// (issue #105) — can be added in place, alongside an ordinary dataset in the
/// same commit.
#[test]
fn add_empty_dataset_via_edit_session() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_empty.h5");
    write_starter(&path);

    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("empty")
            .with_f64_data(&[])
            .with_shape(&[0, 3]);
        session.create_dataset("added").with_i32_data(&[7, 8]);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let empty = file.dataset("empty").unwrap();
    assert_eq!(empty.shape().unwrap(), vec![0, 3]);
    assert_eq!(empty.dtype().unwrap(), DType::F64);
    assert_eq!(empty.read_f64().unwrap(), Vec::<f64>::new());
    assert_eq!(
        file.dataset("added").unwrap().read_i32().unwrap(),
        vec![7, 8]
    );
    // The original, pre-existing dataset is untouched.
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    std::fs::remove_file(&path).ok();
}

/// Chunking a zero-element shape stays refused in place (it's a distinct,
/// separately-tracked capability from the plain contiguous empty dataset
/// above): `malformed_chunked_requests_are_rejected_without_writing` already
/// covers the whole-file-writer-equivalent geometry refusal; this confirms
/// `EditSession` refuses it too, cleanly, without writing anything.
#[test]
fn add_chunked_empty_dataset_is_rejected_without_writing() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_chunked_empty.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("stream")
            .with_i32_data(&[])
            .with_shape(&[0])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[16]);
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("empty"), "got: {err}");
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(&path).ok();
}

/// An empty dataset whose supplied data does not match its (zero-element)
/// shape is rejected rather than silently written as unreachable, orphaned
/// storage: `flatten_dataset` must validate `raw` against the shape even when
/// the shape contains a `0` dimension, not just for non-empty shapes.
#[test]
fn add_empty_dataset_with_mismatched_data_is_rejected_without_writing() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_empty_mismatched.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("bogus")
            .with_f64_data(&[9.0, 9.0, 9.0])
            .with_shape(&[0, 3]);
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("shape"), "got: {err}");
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(&path).ok();
}

/// A provenance-tagged dataset (issue #105) can be added in place; the
/// SHA-256/creator/timestamp/source attributes are computed and stored
/// exactly as the whole-file writer does, and `verify_provenance` confirms
/// the hash while the attribute values themselves are checked directly.
#[cfg(feature = "provenance")]
#[test]
fn add_provenance_dataset_via_edit_session() {
    use hdf5_pure::VerifyResult;

    let path = std::env::temp_dir().join("hdf5_pure_edit_add_provenance.h5");
    write_starter(&path);

    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("sensor")
            .with_f64_data(&[1.0, 2.0, 3.0])
            .with_provenance("test-suite", "2026-02-19T12:00:00Z", Some("bench"));
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("sensor").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(ds.verify_provenance().unwrap(), VerifyResult::Ok);
    let attrs = ds.attrs().unwrap();
    assert_eq!(
        attrs.get("_provenance_creator"),
        Some(&AttrValue::String("test-suite".into()))
    );
    assert_eq!(
        attrs.get("_provenance_timestamp"),
        Some(&AttrValue::String("2026-02-19T12:00:00Z".into()))
    );
    assert_eq!(
        attrs.get("_provenance_source"),
        Some(&AttrValue::String("bench".into()))
    );
    std::fs::remove_file(&path).ok();
}

/// A provenance-tagged dataset can also be chunked/compressed in the same
/// commit: provenance attributes and chunked storage flow through the same
/// `attrs` vec and apply-loop path independently, so this combination should
/// just work — this test is the regression guard for that claim.
#[cfg(all(feature = "provenance", feature = "deflate"))]
#[test]
fn add_provenance_chunked_dataset_via_edit_session() {
    use hdf5_pure::VerifyResult;

    let path = std::env::temp_dir().join("hdf5_pure_edit_add_provenance_chunked.h5");
    write_starter(&path);

    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("sensor_chunked")
            .with_f64_data(&[1.0, 2.0, 3.0, 4.0])
            .with_shape(&[4])
            .with_chunks(&[2])
            .with_deflate(6)
            .with_provenance("test-suite", "2026-02-19T12:00:00Z", None);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("sensor_chunked").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(ds.verify_provenance().unwrap(), VerifyResult::Ok);
    let attrs = ds.attrs().unwrap();
    assert_eq!(
        attrs.get("_provenance_creator"),
        Some(&AttrValue::String("test-suite".into()))
    );
    assert!(
        !attrs.contains_key("_provenance_source"),
        "no source was given, so no source attribute should be written"
    );
    std::fs::remove_file(&path).ok();
}

/// Provenance attributes (up to 4) are appended to `attrs` before the
/// dense-attribute (`MAX_COMPACT_ATTRS` = 8) budget check runs, so a dataset
/// that would otherwise fit can be pushed over the limit by its own
/// provenance metadata. Exactly at the boundary (8 total) must succeed with
/// every value intact.
#[cfg(feature = "provenance")]
#[test]
fn add_provenance_dataset_at_attr_budget_boundary_via_edit_session() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_provenance_at_budget.h5");
    write_starter(&path);

    {
        let mut session = EditSession::open(&path).unwrap();
        let ds = session.create_dataset("sensor_at_budget");
        ds.with_f64_data(&[1.0, 2.0]);
        // 4 plain attributes + 4 provenance attributes (source included) = 8.
        for i in 0..4i64 {
            ds.set_attr(&format!("plain_{i}"), AttrValue::I64(i));
        }
        ds.with_provenance("test-suite", "2026-02-19T12:00:00Z", Some("bench"));
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("sensor_at_budget").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0]);
    let attrs = ds.attrs().unwrap();
    for i in 0..4i64 {
        assert_eq!(
            attrs.get(&format!("plain_{i}")),
            Some(&AttrValue::I64(i)),
            "plain_{i} attribute value mismatch"
        );
    }
    assert_eq!(
        attrs.get("_provenance_creator"),
        Some(&AttrValue::String("test-suite".into()))
    );
    std::fs::remove_file(&path).ok();
}

/// One attribute past the boundary above (9 total, once provenance is
/// appended) is refused, and the refusal must not write anything.
#[cfg(feature = "provenance")]
#[test]
fn add_provenance_dataset_over_attr_budget_is_rejected_without_writing() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_provenance_over_budget.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        let ds = session.create_dataset("sensor_over_budget");
        ds.with_f64_data(&[1.0, 2.0]);
        // 5 plain attributes + 4 provenance attributes (source included) = 9.
        for i in 0..5 {
            ds.set_attr(&format!("plain_{i}"), AttrValue::I32(i));
        }
        ds.with_provenance("test-suite", "2026-02-19T12:00:00Z", Some("bench"));
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("dense"), "got: {err}");
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(&path).ok();
}

/// A dataset with a variable-length attribute (issue #105) can be added in
/// place: its global heap collection is placed and its placeholder reference
/// patched during commit, alongside the dataset's own fixed-size attributes.
#[test]
fn add_dataset_with_variable_length_attribute_via_edit_session() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_dataset_vlen_attr.h5");
    write_starter(&path);

    {
        let mut session = EditSession::open(&path).unwrap();
        let ds = session.create_dataset("labeled");
        ds.with_i32_data(&[1, 2, 3]);
        ds.set_attr(
            "tags",
            AttrValue::VarLenAsciiArray(vec!["one".into(), "two".into(), "three".into()]),
        );
        ds.set_attr("scale", AttrValue::F64(2.5));
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("labeled").unwrap();
    assert_eq!(ds.read_i32().unwrap(), vec![1, 2, 3]);
    let attrs = ds.attrs().unwrap();
    assert_eq!(
        attrs.get("tags"),
        Some(&AttrValue::StringArray(vec![
            "one".into(),
            "two".into(),
            "three".into()
        ]))
    );
    assert_eq!(attrs.get("scale"), Some(&AttrValue::F64(2.5)));
    // The original, pre-existing dataset is untouched.
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    std::fs::remove_file(&path).ok();
}

/// A variable-length attribute is patched before the chunked/non-chunked
/// apply branch, so a chunked dataset can carry one too (VL attributes live
/// in the object header, not inside a chunk — only VL-string *data* is
/// refused when chunked, not VL attributes).
#[test]
fn add_chunked_dataset_with_variable_length_attribute_via_edit_session() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_chunked_vlen_attr.h5");
    write_starter(&path);

    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("chunky_labeled")
            .with_f64_data(&data)
            .with_chunks(&[25])
            .set_attr(
                "tags",
                AttrValue::VarLenAsciiArray(vec!["one".into(), "two".into()]),
            );
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("chunky_labeled").unwrap();
    assert_eq!(ds.read_f64().unwrap(), data);
    assert_eq!(
        ds.attrs().unwrap().get("tags"),
        Some(&AttrValue::StringArray(vec!["one".into(), "two".into()]))
    );
    std::fs::remove_file(&path).ok();
}

/// A dataset attribute whose serialized message overflows the object header's
/// 2-byte message-size field is refused rather than silently truncated (a
/// `VarLenAsciiArray` with enough strings is the practical way to reach this;
/// each string element serializes to a fixed-size global-heap reference, so
/// enough of them push the message past `u16::MAX` bytes).
#[test]
fn add_dataset_with_oversized_variable_length_attribute_is_rejected_without_writing() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_oversized_vlen_attr.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        // Each element serializes to a fixed-size 16-byte global-heap
        // reference; 5000 of them (80000 bytes) comfortably overflows the
        // object header's 2-byte (`u16::MAX` = 65535) message-size field.
        let strings: Vec<String> = (0..5000).map(|i| i.to_string()).collect();
        session
            .create_dataset("oversized")
            .with_i32_data(&[1])
            .set_attr("tags", AttrValue::VarLenAsciiArray(strings));
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("too large"), "got: {err}");
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(&path).ok();
}

/// Regression test for issue #105's silent-corruption bug: a variable-length
/// string dataset (`with_vlen_strings`) added via `EditSession` used to commit
/// `Ok(())` without ever writing its global heap collection or patching its
/// placeholder references, so the dataset failed to read back
/// (`InvalidGlobalHeapSignature`). It must now round-trip like any other
/// added dataset.
#[test]
fn add_vlen_string_dataset_via_edit_session() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_vlen_string_dataset.h5");
    write_starter(&path);

    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("labels")
            .with_vlen_strings(&["alpha", "", "gamma"]);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("labels").unwrap();
    assert_eq!(
        ds.read_string().unwrap(),
        vec!["alpha".to_string(), String::new(), "gamma".to_string()]
    );
    // The original, pre-existing dataset is untouched.
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn add_chunked_vlen_string_dataset_is_rejected_without_writing() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_chunked_vlen_string.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("labels")
            .with_vlen_strings(&["a", "b", "c"])
            .with_chunks(&[2]);
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("variable-length-string"),
            "got: {err}"
        );
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(&path).ok();
}

/// An object-reference dataset (issue #105) can be added in place, targeting
/// an object that already existed before this commit — resolved via the
/// pre-commit-file fallback in `resolve_reference_target` since the target is
/// untouched by this commit.
#[test]
fn add_reference_dataset_targeting_preexisting_object_via_edit_session() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_ref_preexisting.h5");
    write_starter(&path);

    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("refs")
            .with_path_references(&["original"]);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let targets = file.dataset("refs").unwrap().dereference().unwrap();
    assert_eq!(targets.len(), 1);
    match &targets[0] {
        Object::Dataset(ds) => assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]),
        Object::Group(_) => panic!("expected a dataset reference"),
    }
    std::fs::remove_file(&path).ok();
}

/// A reference dataset may target a sibling dataset added in the **same**
/// commit and the **same** group, regardless of which one was staged first —
/// the apply loop places every non-reference dataset in a group before any
/// reference dataset in that group (issue #105).
#[test]
fn add_reference_dataset_targeting_sibling_added_in_same_commit() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_ref_sibling.h5");
    write_starter(&path);

    {
        let mut session = EditSession::open(&path).unwrap();
        // Stage the reference dataset BEFORE its target to prove placement
        // order is independent of `pending_datasets` staging order.
        session
            .create_dataset("refs")
            .with_path_references(&["target"]);
        session.create_dataset("target").with_i32_data(&[7, 8, 9]);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let targets = file.dataset("refs").unwrap().dereference().unwrap();
    assert_eq!(targets.len(), 1);
    match &targets[0] {
        Object::Dataset(ds) => assert_eq!(ds.read_i32().unwrap(), vec![7, 8, 9]),
        Object::Group(_) => panic!("expected a dataset reference"),
    }
    std::fs::remove_file(&path).ok();
}

/// A path with no object anywhere — neither pre-existing nor added in this
/// commit — becomes an undefined reference rather than an error, mirroring
/// the whole-file writer's resolution convention for the same builder type
/// (issue #105).
#[test]
fn add_reference_dataset_targeting_nonexistent_path_becomes_undefined() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_ref_nonexistent.h5");
    write_starter(&path);

    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("refs")
            .with_path_references(&["does/not/exist"]);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let err = file.dataset("refs").unwrap().dereference().unwrap_err();
    assert!(err.to_string().contains("null/undefined"), "got: {err}");
    std::fs::remove_file(&path).ok();
}

/// A reference targeting an **ancestor group of its own dataset** is refused:
/// the ancestor's own address is not known until after all of its children —
/// including this reference dataset — are placed, so resolving it now would
/// require a stale or made-up address (issue #105).
#[test]
fn add_reference_dataset_targeting_unprocessed_ancestor_is_rejected_without_writing() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_ref_ancestor.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_dataset("refs").with_path_references(&[""]); // root, its own parent
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("still writing"), "got: {err}");
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(&path).ok();
}

/// A reference targeting a same-depth sibling **group** that the deepest-first
/// apply order has not reached yet is refused for the same reason as an
/// unprocessed ancestor: `"a"` sorts (and is therefore processed) before
/// `"b"`, so `"b"`'s address is not yet known when `"a/refs"` resolves
/// (issue #105).
#[test]
fn add_reference_dataset_targeting_unprocessed_sibling_group_is_rejected_without_writing() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_ref_sibling_group.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_group("a");
        session.create_group("b");
        session
            .create_dataset("a/refs")
            .with_path_references(&["b"]);
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("still writing"), "got: {err}");
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(&path).ok();
}

#[test]
fn add_chunked_reference_dataset_is_rejected_without_writing() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_chunked_ref.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("refs")
            .with_path_references(&["original"])
            .with_chunks(&[1]);
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("object-reference"), "got: {err}");
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(&path).ok();
}

/// The chunked/extensible variable-length-string refusal is an `||` of two
/// independent conditions (`chunk_options.is_chunked()` and
/// `maxshape.is_some()`); this exercises the `with_maxshape`-alone half,
/// which the test above does not (it only sets `with_chunks`).
#[test]
fn add_extensible_vlen_string_dataset_is_rejected_without_writing() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_extensible_vlen_string.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("labels")
            .with_vlen_strings(&["a", "b", "c"])
            .with_maxshape(&[u64::MAX]);
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("variable-length-string"),
            "got: {err}"
        );
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(&path).ok();
}

/// A zero-element variable-length-string dataset (an empty `with_vlen_strings`
/// call) is a valid degenerate case: no global heap collection needs to be
/// placed at all, and the layout falls back to the same undefined-address
/// sentinel as any other empty dataset.
#[test]
fn add_zero_element_vlen_string_dataset_via_edit_session() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_add_zero_vlen_string.h5");
    write_starter(&path);

    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_dataset("labels").with_vlen_strings(&[]);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("labels").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![0]);
    assert_eq!(ds.read_string().unwrap(), Vec::<String>::new());
    std::fs::remove_file(&path).ok();
}

/// Regression test: `write_dataset(...).with_vlen_strings(...)` must not
/// silently corrupt the target (the overwrite path never patches the global
/// heap collection this stages) — the same bug class issue #105 fixed for the
/// *add* path, reached here through the sibling overwrite API instead. It is
/// refused up front, and the refusal must not write anything.
#[test]
fn write_dataset_rejects_vlen_strings_without_writing() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_write_vlen_string_rejected.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("labels").with_vlen_strings(&["a", "b"]);
    b.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .write_dataset("labels")
            .with_vlen_strings(&["x", "y"]);
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("variable-length-string"),
            "got: {err}"
        );
    }

    // The refusal must not touch the file, and the original data must still
    // read back correctly (not partially patched or corrupted).
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("labels").unwrap().read_string().unwrap(),
        vec!["a".to_string(), "b".to_string()]
    );
    std::fs::remove_file(&path).ok();
}
