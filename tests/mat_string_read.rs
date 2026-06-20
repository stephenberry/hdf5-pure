#![cfg(feature = "serde")]
//! Read-back tests for MATLAB's modern `string` class (MCOS opaque objects).
//!
//! With `Options::with_modern_strings()` the serializer writes `String` values
//! as `mxOPAQUE_CLASS` `string` objects: a `uint32` metadata array on the
//! variable, a `uint64` saveobj payload under `#refs#`, and a `#subsystem#/MCOS`
//! `FileWrapper__` store. These tests round-trip such values through the
//! pure-Rust reader, which resolves the object id against the MCOS store and
//! decodes the UTF-16 saveobj payload. They use only `hdf5_pure` (no reference C
//! library), so they run without cmake.
//!
//! Validation note: this proves the reader inverts *our own writer's* `string`
//! layout (which mirrors MATLAB's). Confirming reads of `string` arrays written
//! by real MATLAB still wants a checked-in MATLAB-produced fixture.

use hdf5_pure::mat::{self, Options};
use serde::{Deserialize, Serialize};

fn modern_string_roundtrip<T>(value: &T) -> T
where
    T: Serialize + serde::de::DeserializeOwned,
{
    let bytes = mat::to_bytes_with_options(value, &Options::with_modern_strings()).expect("ser");
    mat::from_bytes(&bytes).expect("de")
}

#[test]
fn scalar_string_field_roundtrips() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Root {
        greeting: String,
    }
    let root = Root {
        greeting: "hello".into(),
    };
    assert_eq!(modern_string_roundtrip(&root), root);
}

#[test]
fn multiple_string_fields_roundtrip() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Root {
        greeting: String,
        name: String,
        empty: String,
    }
    let root = Root {
        greeting: "hello".into(),
        name: "world!!".into(),
        empty: String::new(),
    };
    assert_eq!(modern_string_roundtrip(&root), root);
}

#[test]
fn string_fields_mixed_with_numbers_roundtrip() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Root {
        label: String,
        trial: u32,
        samples: Vec<f64>,
        note: String,
    }
    let root = Root {
        label: "run-1".into(),
        trial: 7,
        samples: vec![1.0, 2.0, 3.0],
        note: "ok".into(),
    };
    assert_eq!(modern_string_roundtrip(&root), root);
}

#[test]
fn nested_struct_string_fields_roundtrip() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Inner {
        title: String,
        unit: String,
    }
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Root {
        name: String,
        inner: Inner,
    }
    let root = Root {
        name: "outer".into(),
        inner: Inner {
            title: "temperature".into(),
            unit: "°C".into(),
        },
    };
    assert_eq!(modern_string_roundtrip(&root), root);
}

#[test]
fn unicode_and_surrogate_pairs_roundtrip() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Root {
        emoji: String,
        accents: String,
        cjk: String,
    }
    let root = Root {
        // Emoji require UTF-16 surrogate pairs; the saveobj packs raw code units.
        emoji: "🌍🚀😂".into(),
        accents: "naïve café résumé".into(),
        cjk: "日本語テスト".into(),
    };
    assert_eq!(modern_string_roundtrip(&root), root);
}

#[test]
fn vec_of_strings_roundtrips_as_cell_of_string_objects() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Root {
        words: Vec<String>,
    }
    // A `Vec<String>` lowers to a cell whose elements are individual `string`
    // objects; reading resolves each object id against the shared MCOS store.
    let root = Root {
        words: vec!["alpha".into(), "beta".into(), "".into(), "δέλτα".into()],
    };
    assert_eq!(modern_string_roundtrip(&root), root);
}

#[test]
fn unsupported_opaque_class_is_refused_by_name() {
    use hdf5_pure::mat::MatError;
    use hdf5_pure::{AttrValue, FileBuilder};

    // Hand-craft an opaque dataset that claims to be `datetime` (decode = 3),
    // which the reader does not yet decode. It must refuse by name rather than
    // misread or panic. (The metadata shape mirrors a real opaque parent:
    // `[MAGIC, ndims=2, 1, 1, object_id=1, class_id=1]`.)
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("t")
        .with_u32_data(&[0xDD00_0000, 2, 1, 1, 1, 1])
        .with_shape(&[1, 6])
        .set_attr("MATLAB_class", AttrValue::String("datetime".into()))
        .set_attr("MATLAB_object_decode", AttrValue::I32(3));
    let bytes = builder.finish().unwrap();

    #[derive(Deserialize, Debug)]
    struct Root {
        #[allow(dead_code)]
        t: f64,
    }
    let err = mat::from_bytes::<Root>(&bytes).expect_err("datetime is not yet supported");
    assert!(
        matches!(err, MatError::UnsupportedMatlabClass(ref c) if c == "datetime"),
        "expected UnsupportedMatlabClass(\"datetime\"), got: {err}"
    );
}
