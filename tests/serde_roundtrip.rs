#![cfg(feature = "serde")]
//! Roundtrip tests for `.mat` v7.3 serde support.
//!
//! These tests exercise the full serializer path and verify the produced
//! bytes can be read back via the hdf5-pure reader (and will eventually be
//! deserialized back into the original struct).

use hdf5_pure::mat::{self, Complex32, Complex64, Matrix};
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

    // name: char [4, 1] HDF5 → MATLAB row [1, 4] (UTF-16 code units of "beta")
    let name = file.dataset("name").unwrap();
    assert_eq!(name.shape().unwrap(), vec![4, 1]);
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
    assert_eq!(ds.shape().unwrap(), vec![5, 1]);
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
    // MATLAB row-vector string: HDF5 shape [N, 1].
    assert_eq!(ds.shape().unwrap(), vec![2, 1]);
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

// ---------------------------------------------------------------------------
// Edge-case roundtrips
// ---------------------------------------------------------------------------

/// Every integer Matrix<T> must survive roundtrip preserving dtype and values.
#[test]
fn roundtrip_int_matrices() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Ints {
        m_i8: Matrix<i8>,
        m_i16: Matrix<i16>,
        m_i32: Matrix<i32>,
        m_i64: Matrix<i64>,
        m_u8: Matrix<u8>,
        m_u16: Matrix<u16>,
        m_u32: Matrix<u32>,
        m_u64: Matrix<u64>,
    }
    rt(Ints {
        m_i8: Matrix::from_row_major(2, 2, vec![-128_i8, 127, 0, -1]),
        m_i16: Matrix::from_row_major(2, 2, vec![-32768_i16, 32767, 0, -1]),
        m_i32: Matrix::from_row_major(2, 3, vec![-1_i32, 2, 3, 4, 5, 6]),
        m_i64: Matrix::from_row_major(2, 2, vec![i64::MIN, -1_i64, 0, i64::MAX]),
        m_u8: Matrix::from_row_major(2, 2, vec![0_u8, 1, 254, 255]),
        m_u16: Matrix::from_row_major(2, 2, vec![0_u16, 1, 65534, 65535]),
        m_u32: Matrix::from_row_major(2, 2, vec![0_u32, 1, u32::MAX - 1, u32::MAX]),
        m_u64: Matrix::from_row_major(2, 2, vec![0_u64, 1, 2, u64::MAX]),
    });
}

#[test]
fn roundtrip_bool_matrix() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct BoolMat { m: Matrix<bool> }
    rt(BoolMat {
        m: Matrix::from_row_major(3, 3, vec![
            true, false, true,
            false, true, false,
            true, false, true,
        ]),
    });
}

/// 1×1, 1×N, and N×1 matrix orientations for multiple element types.
#[test]
fn roundtrip_matrix_shape_edges() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Shapes {
        m_1x1: Matrix<f64>,
        m_1x5: Matrix<f64>,
        m_5x1: Matrix<f64>,
        i_5x1: Matrix<i32>,
        i_1x5: Matrix<i32>,
        b_3x1: Matrix<bool>,
        b_1x3: Matrix<bool>,
    }
    rt(Shapes {
        m_1x1: Matrix::from_row_major(1, 1, vec![42.0]),
        m_1x5: Matrix::from_row_major(1, 5, vec![10.0, 20.0, 30.0, 40.0, 50.0]),
        m_5x1: Matrix::from_row_major(5, 1, vec![10.0, 20.0, 30.0, 40.0, 50.0]),
        i_5x1: Matrix::from_row_major(5, 1, vec![-1, 0, 1, 2, 3]),
        i_1x5: Matrix::from_row_major(1, 5, vec![-1, 0, 1, 2, 3]),
        b_3x1: Matrix::from_row_major(3, 1, vec![true, false, true]),
        b_1x3: Matrix::from_row_major(1, 3, vec![true, false, true]),
    });
}

/// Bit-exact preservation of IEEE 754 specials: NaN, ±Inf, subnormal, -0.
#[test]
fn roundtrip_f64_bit_exact_specials() {
    #[derive(Serialize, Deserialize)]
    struct Specials {
        nan_bits: f64,
        pos_inf: f64,
        neg_inf: f64,
        neg_zero: f64,
        subnormal: f64,
        vals: Vec<f64>,
    }
    // Specific NaN bit pattern we can check after roundtrip.
    let quiet_nan = f64::from_bits(0x7FF8_0000_DEAD_BEEF);
    let v = Specials {
        nan_bits: quiet_nan,
        pos_inf: f64::INFINITY,
        neg_inf: f64::NEG_INFINITY,
        neg_zero: -0.0,
        subnormal: f64::from_bits(1),
        vals: vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0, 0.0],
    };
    let bytes = mat::to_bytes(&v).unwrap();
    let back: Specials = mat::from_bytes(&bytes).unwrap();
    // NaN payload must survive (quiet_nan bits preserved exactly).
    assert_eq!(back.nan_bits.to_bits(), quiet_nan.to_bits(), "nan payload");
    assert!(back.pos_inf.is_infinite() && back.pos_inf > 0.0);
    assert!(back.neg_inf.is_infinite() && back.neg_inf < 0.0);
    // -0 preserved bit-exact.
    assert_eq!(back.neg_zero.to_bits(), 0x8000_0000_0000_0000);
    // Subnormal preserved bit-exact.
    assert_eq!(back.subnormal.to_bits(), 1);
    // Vec contents.
    assert!(back.vals[0].is_nan());
    assert!(back.vals[1].is_infinite() && back.vals[1] > 0.0);
    assert!(back.vals[2].is_infinite() && back.vals[2] < 0.0);
    assert_eq!(back.vals[3].to_bits(), 0x8000_0000_0000_0000);
    assert_eq!(back.vals[4].to_bits(), 0);
}

/// Integer extremes: i64::MIN/MAX, u64::MAX.
#[test]
fn roundtrip_integer_extremes() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Ex {
        i64_min: i64,
        i64_max: i64,
        u64_max: u64,
        i8_min: i8,
        i8_max: i8,
        vals: Vec<i64>,
    }
    rt(Ex {
        i64_min: i64::MIN,
        i64_max: i64::MAX,
        u64_max: u64::MAX,
        i8_min: i8::MIN,
        i8_max: i8::MAX,
        vals: vec![i64::MIN, -1, 0, 1, i64::MAX],
    });
}

/// Deeply nested struct (4 levels) roundtrip.
#[test]
fn roundtrip_deep_nested() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct L1 { label: String, l2: L2 }
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct L2 { depth: u32, l3: L3 }
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct L3 { tag: String, l4: L4 }
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct L4 { id: u64, values: Vec<f64> }
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Root { root: L1 }
    rt(Root {
        root: L1 {
            label: "top".into(),
            l2: L2 {
                depth: 2,
                l3: L3 {
                    tag: "middle".into(),
                    l4: L4 { id: 999, values: vec![1.5, 2.5, 3.5] },
                },
            },
        },
    });
}

/// Non-BMP Unicode (surrogate pair) roundtrip.
#[test]
fn roundtrip_surrogate_pair() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Emoji { party: String, mixed: String }
    rt(Emoji {
        party: "🎉".into(),
        mixed: "é日🎉A".into(),
    });
}

/// Empty containers of every primitive type survive roundtrip.
#[test]
fn roundtrip_empty_variants() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct E {
        e_f64: Vec<f64>,
        e_f32: Vec<f32>,
        e_i32: Vec<i32>,
        e_u8: Vec<u8>,
        e_bool: Vec<bool>,
        e_str: String,
    }
    rt(E {
        e_f64: vec![],
        e_f32: vec![],
        e_i32: vec![],
        e_u8: vec![],
        e_bool: vec![],
        e_str: String::new(),
    });
}

/// Large matrix roundtrip — catches any off-by-one in transposition at scale.
#[test]
fn roundtrip_large_matrix() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Big { m: Matrix<f64> }
    let rows = 100;
    let cols = 50;
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            data.push((r * 1000 + c) as f64);
        }
    }
    rt(Big { m: Matrix::from_row_major(rows, cols, data) });
}

/// Complex32 scalar and vec survive roundtrip preserving precision.
#[test]
fn roundtrip_complex32() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct C32 {
        z: Complex32,
        v: Vec<Complex32>,
    }
    rt(C32 {
        z: Complex32::new(1.25, -0.5),
        v: vec![
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 1.0),
            Complex32::new(-1.0, -1.0),
        ],
    });
}

/// Bit-exact preservation of NaN/Inf/-0 inside a Matrix (not just scalars).
#[test]
fn roundtrip_matrix_with_specials() {
    #[derive(Serialize, Deserialize)]
    struct MatS { m: Matrix<f64> }
    let m = Matrix::from_row_major(
        2,
        3,
        vec![1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0, 0.0],
    );
    let bytes = mat::to_bytes(&MatS { m }).unwrap();
    let back: MatS = mat::from_bytes(&bytes).unwrap();
    assert_eq!(back.m.rows(), 2);
    assert_eq!(back.m.cols(), 3);
    let d = back.m.data();
    assert_eq!(d[0], 1.0);
    assert!(d[1].is_nan());
    assert!(d[2].is_infinite() && d[2] > 0.0);
    assert!(d[3].is_infinite() && d[3] < 0.0);
    assert_eq!(d[4].to_bits(), 0x8000_0000_0000_0000); // -0
    assert_eq!(d[5].to_bits(), 0); // +0
}

/// Option<Struct> and Option<Vec<T>> work in both Some and None.
#[test]
fn roundtrip_option_complex_inner() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Inner { k: u32, v: f64 }
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Outer {
        some_struct: Option<Inner>,
        none_struct: Option<Inner>,
        some_vec: Option<Vec<f64>>,
        none_vec: Option<Vec<f64>>,
    }
    rt(Outer {
        some_struct: Some(Inner { k: 7, v: 3.14 }),
        none_struct: None,
        some_vec: Some(vec![1.0, 2.0, 3.0]),
        none_vec: None,
    });
}
