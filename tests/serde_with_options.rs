//! Tests for the new `to_bytes_with_options` / `to_file_with_options` API.

use hdf5_pure::mat::{
    self, Compression, EmptyMarkerEncoding, InvalidNamePolicy, Options, StringClass,
};
use hdf5_pure::{AttrValue, File};
use serde::{Deserialize, Serialize};

fn temp_path(name: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("hdf5pure-mat-opts-{name}-{nanos}.mat"))
}

fn read_class(file: &File, ds_path: &str) -> String {
    let ds = file.dataset(ds_path).unwrap();
    let attrs = ds.attrs().unwrap();
    match &attrs["MATLAB_class"] {
        AttrValue::AsciiString(s) | AttrValue::String(s) => s.clone(),
        other => panic!("unexpected class: {other:?}"),
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Doc {
    name: String,
    score: f64,
}

#[test]
fn default_options_produce_char_strings() {
    let doc = Doc {
        name: "alice".into(),
        score: 9.5,
    };
    let bytes = mat::to_bytes_with_options(&doc, &Options::default()).unwrap();
    let path = temp_path("default-char");
    std::fs::write(&path, &bytes).unwrap();
    let f = File::open(&path).unwrap();
    assert_eq!(read_class(&f, "name"), "char");
    assert_eq!(read_class(&f, "score"), "double");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn string_class_option_produces_string_objects() {
    let doc = Doc {
        name: "alice".into(),
        score: 9.5,
    };
    let mut opts = Options::default();
    opts.string_class = StringClass::String;
    let bytes = mat::to_bytes_with_options(&doc, &opts).unwrap();
    let path = temp_path("string-class");
    std::fs::write(&path, &bytes).unwrap();
    let f = File::open(&path).unwrap();
    assert_eq!(read_class(&f, "name"), "string");
    let sub = f.dataset("#subsystem#/MCOS").unwrap();
    let sub_attrs = sub.attrs().unwrap();
    let sub_class = match &sub_attrs["MATLAB_class"] {
        AttrValue::AsciiString(s) | AttrValue::String(s) => s.clone(),
        _ => panic!(),
    };
    assert_eq!(sub_class, "FileWrapper__");
    std::fs::remove_file(path).unwrap();
}

#[derive(Serialize)]
struct WithKeyword {
    end: u32,
}

#[test]
fn sanitize_policy_rewrites_keywords() {
    let doc = WithKeyword { end: 5 };
    let mut opts = Options::default();
    opts.invalid_name_policy = InvalidNamePolicy::Sanitize;
    let bytes = mat::to_bytes_with_options(&doc, &opts).unwrap();
    let path = temp_path("sanitize");
    std::fs::write(&path, &bytes).unwrap();
    let f = File::open(&path).unwrap();
    assert_eq!(read_class(&f, "end_"), "uint32");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn error_policy_rejects_keywords() {
    let doc = WithKeyword { end: 5 };
    let mut opts = Options::default();
    opts.invalid_name_policy = InvalidNamePolicy::Error;
    let err = mat::to_bytes_with_options(&doc, &opts).unwrap_err();
    assert!(err.to_string().contains("invalid MATLAB name"));
}

#[derive(Serialize)]
struct Big {
    payload: Vec<f64>,
}

#[test]
fn deflate_compression_shrinks_repetitive_data() {
    // 1 MB of zeros should compress dramatically.
    let doc = Big {
        payload: vec![0.0; 128 * 1024],
    };
    let plain = mat::to_bytes_with_options(&doc, &Options::default()).unwrap();
    let mut opts = Options::default();
    opts.compression = Compression::Deflate {
        level: 6,
        shuffle: true,
    };
    let compressed = mat::to_bytes_with_options(&doc, &opts).unwrap();
    assert!(
        compressed.len() < plain.len() / 2,
        "compressed {} not less than half of plain {}",
        compressed.len(),
        plain.len()
    );
}

#[test]
fn data_as_dims_empty_marker_encoding() {
    #[derive(Serialize)]
    struct OnlyEmpty {
        v: Vec<f64>,
    }
    let doc = OnlyEmpty { v: Vec::new() };
    let mut opts = Options::default();
    opts.empty_marker_encoding = EmptyMarkerEncoding::DataAsDims;
    let bytes = mat::to_bytes_with_options(&doc, &opts).unwrap();
    let path = temp_path("data-as-dims");
    std::fs::write(&path, &bytes).unwrap();
    let f = File::open(&path).unwrap();
    let ds = f.dataset("v").unwrap();
    let attrs = ds.attrs().unwrap();
    let empty = match &attrs["MATLAB_empty"] {
        AttrValue::U32(v) => *v as u64,
        AttrValue::U64(v) => *v,
        AttrValue::I32(v) => *v as u64,
        other => panic!("unexpected: {other:?}"),
    };
    assert_eq!(empty, 1);
    // data-as-dims encoding stores the dimension vector as the data. Under
    // the default ColumnVector mode an empty `Vec<f64>` has MATLAB shape
    // `[0, 1]`, so the data-as-dims payload is `[0, 1]`.
    let data = ds.read_u64().unwrap();
    assert_eq!(data, vec![0, 1]);
    std::fs::remove_file(path).unwrap();
}
