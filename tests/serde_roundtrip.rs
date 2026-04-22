#![cfg(feature = "serde")]
//! Roundtrip tests for `.mat` v7.3 serde support.
//!
//! These tests exercise the full serializer path and verify the produced
//! bytes can be read back via the hdf5-pure reader (and will eventually be
//! deserialized back into the original struct).

use hdf5_pure::mat::{self, Complex64, Matrix};
use hdf5_pure::{AttrValue, File};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Sanity checks on the raw output — userblock shape, HDF5 structure, MATLAB
// class attributes. These run before the deserializer exists.
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct SimpleRoot {
    trial: u32,
    temperature: f64,
    name: String,
}

#[test]
fn userblock_signature_present() {
    let v = SimpleRoot { trial: 1, temperature: 23.5, name: "alpha".into() };
    let bytes = mat::to_bytes(&v).unwrap();
    assert!(bytes.len() > 512, "file must be at least userblock + HDF5");
    assert_eq!(&bytes[..6], b"MATLAB");
    assert_eq!(&bytes[124..128], &[0x00, 0x02, b'I', b'M']);
}

#[test]
fn serialize_top_level_struct_produces_three_variables() {
    let v = SimpleRoot { trial: 7, temperature: 300.0, name: "beta".into() };
    let bytes = mat::to_bytes(&v).unwrap();

    let file = File::from_bytes(bytes).unwrap();

    // trial: uint32 scalar
    let trial = file.dataset("trial").unwrap();
    assert_eq!(trial.shape().unwrap(), vec![1, 1]);
    assert_eq!(trial.read_u32().unwrap(), vec![7]);
    let trial_attrs = trial.attrs().unwrap();
    assert_eq!(
        trial_attrs.get("MATLAB_class"),
        Some(&AttrValue::String("uint32".into()))
    );

    // temperature: double scalar
    let temp = file.dataset("temperature").unwrap();
    assert_eq!(temp.shape().unwrap(), vec![1, 1]);
    assert_eq!(temp.read_f64().unwrap(), vec![300.0]);
    assert_eq!(
        temp.attrs().unwrap().get("MATLAB_class"),
        Some(&AttrValue::String("double".into()))
    );

    // name: char [1, 4] (UTF-16 code units of "beta")
    let name = file.dataset("name").unwrap();
    assert_eq!(name.shape().unwrap(), vec![1, 4]);
    assert_eq!(
        name.attrs().unwrap().get("MATLAB_class"),
        Some(&AttrValue::String("char".into()))
    );
    let units = name.read_u16().unwrap();
    let s = String::from_utf16(&units).unwrap();
    assert_eq!(s, "beta");
}

#[test]
fn root_must_be_struct_errors_for_primitives() {
    let r = mat::to_bytes(&42u32);
    assert!(matches!(r, Err(mat::MatError::RootMustBeStruct)));
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Vecs {
    xs: Vec<f64>,
    ns: Vec<i32>,
    bytes: Vec<u8>,
}

#[test]
fn vec_fields_produce_row_vectors() {
    let v = Vecs {
        xs: vec![1.0, 2.0, 3.0, 4.0],
        ns: vec![-1, 0, 1],
        bytes: vec![0xDE, 0xAD, 0xBE, 0xEF],
    };
    let file = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();

    let xs = file.dataset("xs").unwrap();
    assert_eq!(xs.shape().unwrap(), vec![1, 4]);
    assert_eq!(xs.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

    let ns = file.dataset("ns").unwrap();
    assert_eq!(ns.shape().unwrap(), vec![1, 3]);
    assert_eq!(ns.read_i32().unwrap(), vec![-1, 0, 1]);

    let bytes = file.dataset("bytes").unwrap();
    assert_eq!(bytes.shape().unwrap(), vec![1, 4]);
    assert_eq!(bytes.read_u8().unwrap(), vec![0xDE, 0xAD, 0xBE, 0xEF]);
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct WithMatrix {
    m: Matrix<f64>,
}

#[test]
fn matrix_field_has_transposed_shape() {
    // 2-row, 3-col Rust matrix
    //   [[1 2 3]
    //    [4 5 6]]
    let v = WithMatrix {
        m: Matrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    };
    let file = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();
    let m = file.dataset("m").unwrap();
    // HDF5 shape is [cols, rows] so MATLAB sees [rows, cols].
    assert_eq!(m.shape().unwrap(), vec![3, 2]);
    // Stored column-major: col0=[1,4], col1=[2,5], col2=[3,6]
    assert_eq!(m.read_f64().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct WithNested {
    x: f64,
    inner: Inner,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Inner {
    y: i32,
    z: Vec<f64>,
}

#[test]
fn nested_struct_becomes_group() {
    let v = WithNested {
        x: 1.5,
        inner: Inner { y: 42, z: vec![10.0, 20.0] },
    };
    let file = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();

    // Outer fields
    assert_eq!(file.dataset("x").unwrap().read_f64().unwrap(), vec![1.5]);

    // Nested group + MATLAB_class + MATLAB_fields
    let inner = file.group("inner").unwrap();
    let attrs = inner.attrs().unwrap();
    assert_eq!(
        attrs.get("MATLAB_class"),
        Some(&AttrValue::String("struct".into()))
    );
    // The reader decodes variable-length ASCII arrays as StringArray.
    let fields = match attrs.get("MATLAB_fields") {
        Some(AttrValue::StringArray(v)) => v.clone(),
        Some(AttrValue::VarLenAsciiArray(v)) => v.clone(),
        other => panic!("expected string array, got {other:?}"),
    };
    assert_eq!(fields, vec!["y".to_string(), "z".to_string()]);

    // Nested datasets exist
    let y = file.dataset("inner/y").unwrap();
    assert_eq!(y.read_i32().unwrap(), vec![42]);
    let z = file.dataset("inner/z").unwrap();
    assert_eq!(z.shape().unwrap(), vec![1, 2]);
    assert_eq!(z.read_f64().unwrap(), vec![10.0, 20.0]);
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct WithOption {
    required: f64,
    #[serde(default)]
    maybe: Option<String>,
}

#[test]
fn option_none_is_omitted() {
    let v = WithOption { required: 1.0, maybe: None };
    let file = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();
    assert!(file.dataset("required").is_ok());
    assert!(file.dataset("maybe").is_err());
}

#[test]
fn option_some_serializes_underlying() {
    let v = WithOption { required: 1.0, maybe: Some("hello".into()) };
    let file = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();
    let ds = file.dataset("maybe").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![1, 5]);
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct WithComplex {
    z: Complex64,
    zs: Vec<Complex64>,
}

#[test]
fn complex_scalar_and_vector() {
    let v = WithComplex {
        z: Complex64::new(1.0, -2.0),
        zs: vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
    };
    let file = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();

    let z = file.dataset("z").unwrap();
    assert_eq!(z.shape().unwrap(), vec![1, 1]);
    // The compound dataset is double class per MATLAB convention.
    assert_eq!(
        z.attrs().unwrap().get("MATLAB_class"),
        Some(&AttrValue::String("double".into()))
    );

    let zs = file.dataset("zs").unwrap();
    assert_eq!(zs.shape().unwrap(), vec![1, 2]);
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
enum Flag { On, Off }

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct WithEnum {
    state: Flag,
}

#[test]
fn unit_enum_variant_becomes_char_string() {
    let v = WithEnum { state: Flag::On };
    let file = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();
    let ds = file.dataset("state").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![1, 2]);
    let units = ds.read_u16().unwrap();
    assert_eq!(String::from_utf16(&units).unwrap(), "On");
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct WithBool {
    flag: bool,
    flags: Vec<bool>,
}

#[test]
fn bool_fields_are_logical_uint8() {
    let v = WithBool { flag: true, flags: vec![true, false, true] };
    let file = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();

    let f = file.dataset("flag").unwrap();
    assert_eq!(f.shape().unwrap(), vec![1, 1]);
    assert_eq!(
        f.attrs().unwrap().get("MATLAB_class"),
        Some(&AttrValue::String("logical".into()))
    );
    assert_eq!(f.read_u8().unwrap(), vec![1]);

    let fs = file.dataset("flags").unwrap();
    assert_eq!(fs.shape().unwrap(), vec![1, 3]);
    assert_eq!(fs.read_u8().unwrap(), vec![1, 0, 1]);
}

// ---------------------------------------------------------------------------
// Full roundtrip: serialize then deserialize back, assert equality.
// ---------------------------------------------------------------------------

fn rt<T: Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug>(value: T) {
    let bytes = mat::to_bytes(&value).expect("serialize");
    let back: T = mat::from_bytes(&bytes).expect("deserialize");
    assert_eq!(back, value);
}

#[test]
fn roundtrip_simple_primitives() {
    rt(SimpleRoot { trial: 9, temperature: 273.15, name: "hello".into() });
}

#[test]
fn roundtrip_vectors() {
    rt(Vecs {
        xs: vec![1.0, 2.0, 3.0, 4.0],
        ns: vec![-1, 0, 1],
        bytes: vec![0xDE, 0xAD, 0xBE, 0xEF],
    });
}

#[test]
fn roundtrip_matrix_f64() {
    rt(WithMatrix {
        m: Matrix::from_row_major(
            3, 4,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ),
    });
}

#[test]
fn roundtrip_nested() {
    rt(WithNested {
        x: 1.5,
        inner: Inner { y: 42, z: vec![10.0, 20.0, 30.0] },
    });
}

#[test]
fn roundtrip_option_some_and_none() {
    rt(WithOption { required: 1.0, maybe: Some("world".into()) });
    rt(WithOption { required: 2.5, maybe: None });
}

#[test]
fn roundtrip_complex_scalar_and_vec() {
    rt(WithComplex {
        z: Complex64::new(2.0, -1.0),
        zs: vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0), Complex64::new(-1.0, -1.0)],
    });
}

#[test]
fn roundtrip_unit_enum() {
    rt(WithEnum { state: Flag::On });
    rt(WithEnum { state: Flag::Off });
}

#[test]
fn roundtrip_bool() {
    rt(WithBool { flag: true, flags: vec![false, true, false, true] });
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct WithStringUnicode {
    tag: String,
}

#[test]
fn roundtrip_non_ascii_string() {
    rt(WithStringUnicode { tag: "résumé 💡 αβγ".into() });
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct VecOfVec {
    grid: Vec<Vec<f64>>,
}

#[test]
fn roundtrip_vec_of_vec_as_matrix() {
    rt(VecOfVec {
        grid: vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ],
    });
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct ComplexMat {
    m: Vec<Vec<Complex64>>,
}

#[test]
fn roundtrip_complex_matrix() {
    rt(ComplexMat {
        m: vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
            vec![Complex64::new(-1.0, 0.0), Complex64::new(0.0, -1.0)],
        ],
    });
}
