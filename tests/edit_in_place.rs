//! Tests for in-place editing via `EditSession` (issue #32, Group C milestone 1).

use hdf5_pure::{DType, EditSession, File, FileBuilder};

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
fn chunked_dataset_is_rejected_without_writing() {
    let path = std::env::temp_dir().join("hdf5_pure_edit_reject_chunked.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("chunky")
            .with_f64_data(&[1.0, 2.0, 3.0, 4.0])
            .with_chunks(&[2]);
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("in-place edit"),
            "unexpected error: {err}"
        );
    }

    // The guard runs before any write, so the file is untouched.
    let after = std::fs::read(&path).unwrap();
    assert_eq!(before, after);
    std::fs::remove_file(&path).ok();
}
