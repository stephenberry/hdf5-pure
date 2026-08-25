//! By-name group lookup: what opens as a group, and what is refused (issue
//! #352).
//!
//! `File::group` and `Group::group` classify the object they resolve to, the way
//! `H5Gopen` does, rather than handing back a `Group` for whatever the name
//! reached. That check reads the object header, so it is only as good as the set
//! of header messages that count as a group: every form this crate can read has
//! to open, or a fix for a wrong-kind lookup becomes a refusal of a right one.

use hdf5_pure::{Error, File, FileBuilder};

/// The two group forms are recognised by different header messages — a v1 group
/// by its symbol table, a v2 group by its link info — and both must open by
/// name, through both lookups.
///
/// The v1 case is the one this guards. Nothing else in the suite opens a
/// symbol-table group *by name*; the walks reach one by enumerating its parent,
/// which classifies against the same predicate but does not fail when it says
/// no — it just leaves the group out. A classifier that had forgotten v1 would
/// refuse every group in every file written in the pre-1.8 format, and no other
/// test would say so.
///
/// Compact and dense v2 link storage are not separate cases here: a v2 group
/// carries a link-info message either way, and it is that message the classifier
/// reads. The storage split is exercised by `fractal_heap_dense_links.rs`.
#[test]
fn both_group_forms_open_by_name() {
    // (form, fixture, a subgroup of the root, that subgroup's own children)
    let cases = [
        ("v1 symbol table", "two_groups.h5", "group1", 1usize),
        ("v2 link info", "v2_groups.h5", "sensors", 2),
    ];

    for (form, fixture, name, children) in cases {
        let file = File::open(format!("tests/fixtures/{fixture}")).unwrap();

        let by_path = file.group(name);
        assert!(
            by_path.is_ok(),
            "{form}: File::group refused {name}: {:?}",
            by_path.err()
        );
        let by_name = file.root().group(name);
        assert!(
            by_name.is_ok(),
            "{form}: Group::group refused {name}: {:?}",
            by_name.err()
        );

        // Opened, and opened onto the right object: a lookup that classified
        // correctly but memoized the wrong address would list the wrong members.
        for (lookup, group) in [
            ("File::group", by_path.unwrap()),
            ("Group::group", by_name.unwrap()),
        ] {
            assert_eq!(
                group.datasets().unwrap().len() + group.groups().unwrap().len(),
                children,
                "{form}: {lookup} opened {name} onto the wrong object"
            );
        }
    }
}

/// The issue's own case, through the public API: a name that resolves to a
/// dataset is not a group, and says so at the lookup.
///
/// Before this, the lookup returned a `Group` whose `attrs()` answered with the
/// dataset's attributes — a wrong answer, not an error — and whose `datasets()`
/// failed later with a `PathNotFound` naming no path.
#[test]
fn a_name_that_reaches_a_dataset_is_refused() {
    let mut b = FileBuilder::new();
    b.create_dataset("x").with_i32_data(&[1]);
    let mut g = b.create_group("g");
    g.create_dataset("y").with_i32_data(&[2]);
    b.add_group(g.finish());
    let file = File::from_bytes(b.finish().unwrap()).unwrap();

    for (label, got) in [
        ("File::group at the root", file.group("x")),
        ("File::group nested", file.group("g/y")),
        ("Group::group at the root", file.root().group("x")),
        ("Group::group nested", file.group("g").unwrap().group("y")),
    ] {
        assert!(
            matches!(got, Err(Error::NotAGroup(_))),
            "{label} answered {:?}",
            got.map(|_| "a group")
        );
    }

    // A missing name stays a missing name. The two failures have to stay
    // distinguishable: a caller choosing between "create it" and "you meant the
    // dataset" is choosing on exactly that difference.
    assert!(matches!(file.group("absent"), Err(Error::Format(_))));
    assert!(matches!(file.root().group("absent"), Err(Error::Format(_))));
}
