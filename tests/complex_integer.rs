#![cfg(feature = "serde")]
//! Complex arrays with integer components.
//!
//! Hardware that samples in the complex plane — ADCs, digitizer front ends —
//! produces pairs of signed 16-bit integers, and widening them to `single` on
//! the way into a `.mat` file doubles the payload while adding nothing: the
//! mantissa bits are zero by construction. These tests pin the layout that
//! avoids that (a `{real, imag}` compound of the component class, with
//! `MATLAB_class` naming the component), the exact-width guarantee that makes
//! the file self-describing, and the shapes and edge values the float complex
//! path already handles.

use hdf5_pure::mat::options::{Compression, OneDimensionalMode};
use hdf5_pure::mat::{
    self, Complex32, Complex64, ComplexI8, ComplexI16, ComplexI32, ComplexI64, ComplexU8,
    ComplexU16, ComplexU32, ComplexU64, Matrix, Options,
};
use hdf5_pure::{AttrValue, Datatype, DatatypeByteOrder, File};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Shapes {
    scalar: ComplexI16,
    row: Vec<ComplexI16>,
    column: Matrix<ComplexI16>,
    matrix: Matrix<ComplexI16>,
    empty: Matrix<ComplexI16>,
}

fn shapes() -> Shapes {
    Shapes {
        scalar: ComplexI16::new(7, -9),
        row: vec![
            ComplexI16::new(1, 2),
            ComplexI16::new(3, 4),
            ComplexI16::new(5, 6),
        ],
        column: Matrix::from_row_major(
            3,
            1,
            vec![
                ComplexI16::new(10, -10),
                ComplexI16::new(20, -20),
                ComplexI16::new(30, -30),
            ],
        ),
        matrix: Matrix::from_row_major(
            2,
            3,
            vec![
                ComplexI16::new(1, -1),
                ComplexI16::new(2, -2),
                ComplexI16::new(3, -3),
                ComplexI16::new(4, -4),
                ComplexI16::new(5, -5),
                ComplexI16::new(6, -6),
            ],
        ),
        empty: Matrix::from_row_major(0, 0, Vec::new()),
    }
}

#[test]
fn every_shape_round_trips() {
    let v = shapes();
    let bytes = mat::to_bytes(&v).unwrap();
    let back: Shapes = mat::from_bytes(&bytes).unwrap();
    assert_eq!(back, v);
}

/// A 1×N and an N×1 array of the same values are different MATLAB arrays, and
/// the shape has to survive the column-major storage round trip in both
/// orientations.
#[test]
fn row_and_column_orientation_survive() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Oriented {
        row: Matrix<ComplexI16>,
        column: Matrix<ComplexI16>,
    }

    let pairs = || {
        vec![
            ComplexI16::new(10, -10),
            ComplexI16::new(20, -20),
            ComplexI16::new(30, -30),
        ]
    };
    let v = Oriented {
        row: Matrix::from_row_major(1, 3, pairs()),
        column: Matrix::from_row_major(3, 1, pairs()),
    };

    let back: Oriented = mat::from_bytes(&mat::to_bytes(&v).unwrap()).unwrap();
    assert_eq!((back.row.rows(), back.row.cols()), (1, 3));
    assert_eq!((back.column.rows(), back.column.cols()), (3, 1));
    assert_eq!(back, v);
}

/// A one-element complex array reads back as a complex *scalar*, which still
/// has to satisfy a `Vec` target — the same allowance the real numeric path
/// makes for a one-element numeric array.
#[test]
fn a_one_element_array_still_deserializes_as_a_vec() {
    let one = Capture {
        samples: vec![ComplexI16::new(5, -5)],
    };
    let back: Capture = mat::from_bytes(&mat::to_bytes(&one).unwrap()).unwrap();
    assert_eq!(back, one);
}

/// An empty complex array carries no pairs, so only the class attribute and
/// the compound datatype can say what it was — which is exactly what a reader
/// needs to hand it back as an empty `int16` complex array rather than an
/// untyped empty.
#[test]
fn an_empty_array_keeps_its_component_class() {
    let v = shapes();
    let bytes = mat::to_bytes(&v).unwrap();
    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("empty").unwrap();

    assert_eq!(
        ds.attrs().unwrap().get("MATLAB_class"),
        Some(&AttrValue::String("int16".into()))
    );
    assert_compound_of(&ds.datatype().unwrap(), 2, true);
    assert!(ds.read_u8().unwrap().is_empty());
    // No `MATLAB_empty` marker: a zero-element compound is already an empty
    // complex array to MATLAB, and the marker would make it an empty double.
    assert!(!ds.attrs().unwrap().contains_key("MATLAB_empty"));
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

/// Assert a `{real, imag}` compound whose components are `size` bytes wide,
/// `real` at offset 0 and `imag` at `size`.
fn assert_compound_of(dt: &Datatype, size: u32, signed: bool) {
    let Datatype::Compound {
        size: total,
        members,
    } = dt
    else {
        panic!("expected a compound datatype, got {dt:?}");
    };
    assert_eq!(*total, size * 2, "compound is two components wide");
    assert_eq!(members.len(), 2);
    assert_eq!(members[0].name, "real");
    assert_eq!(members[0].byte_offset, 0);
    assert_eq!(members[1].name, "imag");
    assert_eq!(members[1].byte_offset, u64::from(size));
    for m in members {
        match &m.datatype {
            Datatype::FixedPoint {
                size: component,
                byte_order,
                signed: is_signed,
                bit_offset,
                bit_precision,
            } => {
                assert_eq!(*component, size);
                assert_eq!(*byte_order, DatatypeByteOrder::LittleEndian);
                assert_eq!(*is_signed, signed);
                assert_eq!(*bit_offset, 0);
                assert_eq!(u32::from(*bit_precision), size * 8);
            }
            other => panic!("expected a fixed-point component, got {other:?}"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Capture {
    samples: Vec<ComplexI16>,
}

/// The datatype is what MATLAB reads the class off, so asserting the round
/// trip alone would accept a file this crate is happy with and MATLAB refuses.
#[test]
fn the_datatype_is_a_four_byte_int16_compound() {
    let c = Capture {
        samples: vec![ComplexI16::new(1, -1), ComplexI16::new(2, -2)],
    };
    let bytes = mat::to_bytes(&c).unwrap();
    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("samples").unwrap();

    assert_compound_of(&ds.datatype().unwrap(), 2, true);
    assert_eq!(
        ds.attrs().unwrap().get("MATLAB_class"),
        Some(&AttrValue::String("int16".into()))
    );
}

/// Halving the payload is the entire point of the feature, so it is pinned by
/// a test rather than assumed from the datatype.
#[test]
fn the_payload_is_four_bytes_per_element_half_of_the_single_equivalent() {
    #[derive(Serialize)]
    struct AsSingle {
        samples: Vec<Complex32>,
    }

    let n = 64;
    let ints = Capture {
        samples: (0..n)
            .map(|i| ComplexI16::new(i as i16, -(i as i16)))
            .collect(),
    };
    let floats = AsSingle {
        samples: (0..n)
            .map(|i| Complex32::new(i as f32, -(i as f32)))
            .collect(),
    };

    let int_payload = payload_len(mat::to_bytes(&ints).unwrap(), "samples");
    let float_payload = payload_len(mat::to_bytes(&floats).unwrap(), "samples");

    assert_eq!(int_payload, 4 * n as usize);
    assert_eq!(float_payload, 8 * n as usize);
}

fn payload_len(bytes: Vec<u8>, name: &str) -> usize {
    File::from_bytes(bytes)
        .unwrap()
        .dataset(name)
        .unwrap()
        .read_u8()
        .unwrap()
        .len()
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Edges {
    i8s: Vec<ComplexI8>,
    i16s: Vec<ComplexI16>,
    i32s: Vec<ComplexI32>,
    i64s: Vec<ComplexI64>,
    u8s: Vec<ComplexU8>,
    u16s: Vec<ComplexU16>,
    u32s: Vec<ComplexU32>,
    u64s: Vec<ComplexU64>,
}

/// Every width's extremes, including the minima that a naive negation or a
/// signed/unsigned mix-up would mangle.
#[test]
fn extreme_component_values_survive() {
    let v = Edges {
        i8s: vec![
            ComplexI8::new(i8::MIN, i8::MAX),
            ComplexI8::new(0, -1),
            ComplexI8::new(-1, 0),
        ],
        i16s: vec![
            ComplexI16::new(i16::MIN, i16::MAX),
            ComplexI16::new(0, 0),
            ComplexI16::new(-1, 1),
            ComplexI16::new(i16::MAX, i16::MIN),
        ],
        i32s: vec![ComplexI32::new(i32::MIN, i32::MAX)],
        i64s: vec![ComplexI64::new(i64::MIN, i64::MAX)],
        u8s: vec![ComplexU8::new(0, u8::MAX)],
        u16s: vec![ComplexU16::new(u16::MAX, 0)],
        u32s: vec![ComplexU32::new(u32::MAX, 1)],
        u64s: vec![ComplexU64::new(u64::MAX, 0)],
    };
    let bytes = mat::to_bytes(&v).unwrap();
    let back: Edges = mat::from_bytes(&bytes).unwrap();
    assert_eq!(back, v);
}

/// Each width reports its own MATLAB class and its own component width; a
/// shared code path that got the class from the wrong place would still round
/// trip through this crate and still be wrong in MATLAB.
#[test]
fn each_width_reports_its_own_class_and_size() {
    let v = Edges {
        i8s: vec![ComplexI8::new(1, 2)],
        i16s: vec![ComplexI16::new(1, 2)],
        i32s: vec![ComplexI32::new(1, 2)],
        i64s: vec![ComplexI64::new(1, 2)],
        u8s: vec![ComplexU8::new(1, 2)],
        u16s: vec![ComplexU16::new(1, 2)],
        u32s: vec![ComplexU32::new(1, 2)],
        u64s: vec![ComplexU64::new(1, 2)],
    };
    let file = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();

    for (name, class, size, signed) in [
        ("i8s", "int8", 1u32, true),
        ("i16s", "int16", 2, true),
        ("i32s", "int32", 4, true),
        ("i64s", "int64", 8, true),
        ("u8s", "uint8", 1, false),
        ("u16s", "uint16", 2, false),
        ("u32s", "uint32", 4, false),
        ("u64s", "uint64", 8, false),
    ] {
        let ds = file.dataset(name).unwrap();
        assert_eq!(
            ds.attrs().unwrap().get("MATLAB_class"),
            Some(&AttrValue::String(class.into())),
            "class of {name}"
        );
        assert_compound_of(&ds.datatype().unwrap(), size, signed);
        assert_eq!(
            ds.read_u8().unwrap().len(),
            2 * size as usize,
            "payload of {name}"
        );
    }
}

// ---------------------------------------------------------------------------
// No silent width changes, in either direction
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct AsI16 {
    samples: Vec<ComplexI16>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct AsF64 {
    samples: Vec<Complex64>,
}

/// Reading an `int16` capture as a float complex would be lossless and is
/// still refused: the component width is part of what the file says it holds,
/// so a Rust type that quietly disagrees with it is a worse witness than an
/// error.
#[test]
fn an_integer_capture_does_not_deserialize_as_float_complex() {
    let bytes = mat::to_bytes(&AsI16 {
        samples: vec![ComplexI16::new(3, -4)],
    })
    .unwrap();
    let err = mat::from_bytes::<AsF64>(&bytes).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("complex double"),
        "error should name the class it wanted: {msg}"
    );
}

/// The lossy direction is refused for the same reason, and would truncate.
#[test]
fn a_float_capture_does_not_deserialize_as_integer_complex() {
    let bytes = mat::to_bytes(&AsF64 {
        samples: vec![Complex64::new(3.5, -4.5)],
    })
    .unwrap();
    let err = mat::from_bytes::<AsI16>(&bytes).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("complex int16"),
        "error should name the class it wanted: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Compression
// ---------------------------------------------------------------------------

/// The deflate path is shape-sensitive, and an empty array must stay
/// unfiltered (a filtered zero-element dataset is what MATLAB chokes on).
#[test]
fn compression_round_trips_and_leaves_the_empty_array_alone() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Both {
        samples: Vec<ComplexI16>,
        empty: Matrix<ComplexI16>,
    }

    let v = Both {
        // Highly compressible: a repeating ramp.
        samples: (0..4096)
            .map(|i| ComplexI16::new((i % 8) as i16, -((i % 8) as i16)))
            .collect(),
        empty: Matrix::from_row_major(0, 0, Vec::new()),
    };

    let plain = mat::to_bytes_with_options(&v, &Options::default()).unwrap();
    let mut opts = Options::default();
    opts.compression = Compression::Deflate {
        level: 6,
        shuffle: false,
    };
    let deflated = mat::to_bytes_with_options(&v, &opts).unwrap();

    assert!(
        deflated.len() < plain.len(),
        "deflate should shrink a repeating ramp: {} vs {}",
        deflated.len(),
        plain.len()
    );
    let back: Both = mat::from_bytes(&deflated).unwrap();
    assert_eq!(back, v);
}

/// A complex vector is a vector, so it takes the configured orientation like
/// every other one. It used to be written as a MATLAB row regardless, which
/// made it the only kind of 1-D array whose shape contradicted both the option
/// and the default write path.
#[test]
fn a_complex_vector_takes_the_configured_orientation_like_a_real_one() {
    #[derive(Serialize)]
    struct Both {
        reals: Vec<i16>,
        cplx: Vec<ComplexI16>,
    }

    let v = Both {
        reals: vec![1, 2, 3],
        cplx: vec![
            ComplexI16::new(1, -1),
            ComplexI16::new(2, -2),
            ComplexI16::new(3, -3),
        ],
    };
    let shapes = |opts: &Options| {
        let file = File::from_bytes(mat::to_bytes_with_options(&v, opts).unwrap()).unwrap();
        (
            file.dataset("reals").unwrap().shape().unwrap(),
            file.dataset("cplx").unwrap().shape().unwrap(),
        )
    };

    let (reals, cplx) = shapes(&Options::default());
    assert_eq!(reals, cplx, "column-vector default");

    let mut row = Options::default();
    row.one_dimensional_mode = OneDimensionalMode::RowVector;
    let (reals_row, cplx_row) = shapes(&row);
    assert_eq!(reals_row, cplx_row, "row-vector mode");
    assert_ne!(reals, reals_row, "the option has to move the shape at all");

    // And the default options path agrees with `to_bytes`, which has no
    // options to consult.
    let plain = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();
    assert_eq!(plain.dataset("cplx").unwrap().shape().unwrap(), cplx);
}

// ---------------------------------------------------------------------------
// The mid-level builder
// ---------------------------------------------------------------------------

#[test]
fn the_builder_writes_complex_integers_in_every_scope() {
    use hdf5_pure::mat::MatBuilder;

    let mut mb = MatBuilder::new(Options::default());
    mb.write_complex_i16("top", &[1, 2], &[(1, -1), (2, -2)])
        .unwrap();
    mb.struct_("nested", |sw| {
        sw.write_complex_i32("field", &[1, 1], &[(7, -7)])?;
        Ok(())
    })
    .unwrap();
    mb.cell("cell", &[1, 1], |cw| {
        cw.push_complex_u8(&[1, 2], &[(1, 2), (3, 4)])?;
        Ok(())
    })
    .unwrap();
    let bytes = mb.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    assert_compound_of(&file.dataset("top").unwrap().datatype().unwrap(), 2, true);
    assert_compound_of(
        &file.dataset("nested/field").unwrap().datatype().unwrap(),
        4,
        true,
    );
    assert_eq!(
        file.dataset("nested/field")
            .unwrap()
            .attrs()
            .unwrap()
            .get("MATLAB_class"),
        Some(&AttrValue::String("int32".into()))
    );
    // The cell element lives under `#refs#`; the first (and only) one here.
    let refs = file.group("#refs#").unwrap();
    let name = refs.datasets().unwrap().into_iter().next().unwrap();
    let ds = refs.dataset(&name).unwrap();
    assert_compound_of(&ds.datatype().unwrap(), 1, false);
    assert_eq!(
        ds.attrs().unwrap().get("MATLAB_class"),
        Some(&AttrValue::String("uint8".into()))
    );
}
