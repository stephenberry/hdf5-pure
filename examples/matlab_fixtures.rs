//! Produce a set of `.mat` v7.3 files for manual verification in MATLAB.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example matlab_fixtures --features serde
//! ```
//!
//! Files are written to `./matlab_fixtures/` and printed to stdout with the
//! MATLAB commands that should verify each. Load them in MATLAB with:
//!
//! ```matlab
//! cd matlab_fixtures
//! % Then run the commands printed by the example.
//! ```

use hdf5_pure::mat::{self, Complex32, Complex64, Matrix};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

fn main() {
    let out = PathBuf::from("matlab_fixtures");
    std::fs::create_dir_all(&out).expect("create output dir");

    write_scalars(&out);
    write_vectors(&out);
    write_matrix(&out);
    write_strings(&out);
    write_nested_struct(&out);
    write_options(&out);
    write_complex(&out);
    write_unit_enum(&out);
    write_everything(&out);

    // Edge-case stress fixtures.
    write_extremes(&out);
    write_shapes(&out);
    write_int_matrices(&out);
    write_unicode(&out);
    write_complex_edges(&out);
    write_deep_nested(&out);
    write_bool_ext(&out);
    write_empty_variants(&out);
    write_large_matrix(&out);

    // Cell-array fixtures (Vec<Struct>, Vec<Option<T>> with None, nested).
    write_cells(&out);

    copy_octave_helpers(&out);

    println!();
    println!("All fixtures written to: {}", out.display());
    println!();
    println!("Quick check in MATLAB:");
    println!("  >> cd('matlab_fixtures')");
    println!("  >> verify                     % runs all assertions");
    println!();
    println!("Or from a shell (GNU Octave):");
    println!("  $ cd matlab_fixtures && octave --no-gui --eval verify");
}

/// Copy `verify.m` and `ok.m` from `examples/octave/` next to the fixture
/// files. Users run `verify` in MATLAB/Octave after `cd`ing into the output
/// directory — the helper scripts need to be on the path alongside.
fn copy_octave_helpers(out: &Path) {
    let src = Path::new("examples/octave");
    for name in ["verify.m", "ok.m"] {
        let from = src.join(name);
        if from.exists() {
            let _ = std::fs::copy(&from, out.join(name));
        } else {
            eprintln!("warning: {} not found; skipping", from.display());
        }
    }
}

// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct Scalars {
    x_f64: f64,
    y_f32: f32,
    n_i32: i32,
    m_i64: i64,
    u_u32: u32,
    v_u8: u8,
    b_true: bool,
    b_false: bool,
}

fn write_scalars(dir: &Path) {
    announce(
        "scalars.mat",
        &[
            "assert(x_f64 == 3.14159, 'x_f64')",
            "assert(y_f32 == single(2.718), 'y_f32')",
            "assert(n_i32 == int32(-42), 'n_i32')",
            "assert(m_i64 == int64(9999999999), 'm_i64')",
            "assert(u_u32 == uint32(2147483648), 'u_u32')",
            "assert(v_u8 == uint8(255), 'v_u8')",
            "assert(b_true == true, 'b_true')",
            "assert(b_false == false, 'b_false')",
            "disp('scalars.mat OK')",
        ],
    );
    let v = Scalars {
        x_f64: 3.14159,
        y_f32: 2.718,
        n_i32: -42,
        m_i64: 9_999_999_999,
        u_u32: 2_147_483_648,
        v_u8: 255,
        b_true: true,
        b_false: false,
    };
    mat::to_file(&v, dir.join("scalars.mat")).unwrap();
    println!("wrote: scalars.mat");
}

// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct Vectors {
    xs: Vec<f64>,
    ns: Vec<i32>,
    flags: Vec<bool>,
    empty: Vec<f64>,
}

fn write_vectors(dir: &Path) {
    announce(
        "vectors.mat",
        &[
            "% MATLAB stores our row-vector Vec<T> as an Nx1 column vector.",
            "assert(isequal(xs, [1.0; 2.0; 3.0; 4.0; 5.0]), 'xs')",
            "assert(isequal(ns, int32([-1; 0; 1])), 'ns')",
            "assert(isequal(flags, logical([1; 0; 1; 1; 0])), 'flags')",
            "assert(isempty(empty), 'empty')",
            "disp('vectors.mat OK')",
        ],
    );
    let v = Vectors {
        xs: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        ns: vec![-1, 0, 1],
        flags: vec![true, false, true, true, false],
        empty: vec![],
    };
    mat::to_file(&v, dir.join("vectors.mat")).unwrap();
    println!("wrote: vectors.mat");
}

// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct Matrices {
    /// 3×4 row-major matrix that should display in MATLAB as a 3×4 matrix
    /// with the expected row/column values.
    a: Matrix<f64>,
    /// 2×2 identity.
    id: Matrix<f64>,
}

fn write_matrix(dir: &Path) {
    announce(
        "matrix.mat",
        &[
            "% Should load as a 3x4 matrix with values laid out row-by-row.",
            "expected = [1 2 3 4; 5 6 7 8; 9 10 11 12];",
            "assert(isequal(a, expected), 'a')",
            "assert(isequal(id, eye(2)), 'id')",
            "disp('matrix.mat OK')",
        ],
    );
    let a = Matrix::from_row_major(
        3,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let id = Matrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    let v = Matrices { a, id };
    mat::to_file(&v, dir.join("matrix.mat")).unwrap();
    println!("wrote: matrix.mat");
}

// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct Strings {
    ascii: String,
    unicode: String,
    empty: String,
}

fn write_strings(dir: &Path) {
    announce(
        "strings.mat",
        &[
            "% UTF-16 char arrays — MATLAB's native string type.",
            "assert(strcmp(ascii, 'hello MATLAB'), 'ascii')",
            "assert(strcmp(unicode, char([233 169 8364])), 'unicode (é ù €)')",
            "% or: assert(strcmp(unicode, 'é ù €'))  % in a UTF-8 editor",
            "assert(isempty(empty), 'empty')",
            "disp('strings.mat OK')",
        ],
    );
    let v = Strings {
        ascii: "hello MATLAB".into(),
        unicode: "é ù €".into(),
        empty: "".into(),
    };
    mat::to_file(&v, dir.join("strings.mat")).unwrap();
    println!("wrote: strings.mat");
}

// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct Experiment {
    name: String,
    trial: u32,
    timestamp: f64,
    config: Config,
    samples: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
struct Config {
    threshold: f64,
    tag: String,
    max_iter: u32,
}

fn write_nested_struct(dir: &Path) {
    announce(
        "nested.mat",
        &[
            "% Nested struct. MATLAB displays with `disp(e)` and `e.config`.",
            "assert(strcmp(e.name, 'run_alpha'), 'name')",
            "assert(e.trial == uint32(7), 'trial')",
            "assert(abs(e.timestamp - 1.7e9) < 1, 'timestamp')",
            "assert(isstruct(e.config), 'config is struct')",
            "assert(strcmp(e.config.tag, 'prod'), 'config.tag')",
            "assert(e.config.threshold == 0.85, 'config.threshold')",
            "assert(e.config.max_iter == uint32(1000), 'config.max_iter')",
            "assert(isequal(e.samples, [10.0; 20.0; 30.0; 40.0]), 'samples')",
            "disp('nested.mat OK')",
        ],
    );
    let v = Experiment {
        name: "run_alpha".into(),
        trial: 7,
        timestamp: 1_700_000_000.0,
        config: Config {
            threshold: 0.85,
            tag: "prod".into(),
            max_iter: 1000,
        },
        samples: vec![10.0, 20.0, 30.0, 40.0],
    };
    // Wrap in a struct so the file has one top-level variable named `e`.
    #[derive(Serialize)]
    struct Wrap<'a> {
        e: &'a Experiment,
    }
    mat::to_file(&Wrap { e: &v }, dir.join("nested.mat")).unwrap();
    println!("wrote: nested.mat");
}

// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct Optional {
    required: f64,
    present: Option<String>,
    absent: Option<String>,
}

fn write_options(dir: &Path) {
    announce(
        "options.mat",
        &[
            "% `absent` is None in Rust → field is not written to the file.",
            "vars = who;",
            "assert(ismember('required', vars), 'required present')",
            "assert(ismember('present', vars), 'present present')",
            "assert(~ismember('absent', vars), 'absent should be missing')",
            "assert(required == 1.5, 'required')",
            "assert(strcmp(present, 'yes'), 'present')",
            "disp('options.mat OK')",
        ],
    );
    let v = Optional {
        required: 1.5,
        present: Some("yes".into()),
        absent: None,
    };
    mat::to_file(&v, dir.join("options.mat")).unwrap();
    println!("wrote: options.mat");
}

// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct ComplexData {
    z: Complex64,
    signal: Vec<Complex64>,
}

fn write_complex(dir: &Path) {
    announce(
        "complex.mat",
        &[
            "assert(iscomplex(z), 'z is complex')",
            "assert(z == complex(1.0, -2.0), 'z value')",
            "assert(isequal(signal, [complex(1,0); complex(0,1); complex(-1,0); complex(0,-1)]), 'signal')",
            "disp('complex.mat OK')",
        ],
    );
    let v = ComplexData {
        z: Complex64::new(1.0, -2.0),
        signal: vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, -1.0),
        ],
    };
    mat::to_file(&v, dir.join("complex.mat")).unwrap();
    println!("wrote: complex.mat");
}

// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
enum Phase {
    Idle,
    Running,
    Done,
}

#[derive(Serialize, Deserialize)]
struct UnitEnum {
    phase: Phase,
}

fn write_unit_enum(dir: &Path) {
    announce(
        "enum.mat",
        &[
            "% Unit enum variants serialize as MATLAB char strings.",
            "assert(strcmp(phase, 'Running'), 'phase')",
            "disp('enum.mat OK')",
        ],
    );
    let v = UnitEnum {
        phase: Phase::Running,
    };
    mat::to_file(&v, dir.join("enum.mat")).unwrap();
    println!("wrote: enum.mat");
}

// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct Everything {
    name: String,
    pi: f64,
    trial: u32,
    active: bool,
    samples: Vec<f64>,
    result: Matrix<f64>,
    signal: Vec<Complex64>,
    phase: Phase,
    config: Config,
    note: Option<String>,
    skipped: Option<String>,
}

fn write_everything(dir: &Path) {
    announce(
        "experiment.mat",
        &[
            "% A realistic struct mixing every supported feature.",
            "assert(strcmp(name, 'full_run'), 'name')",
            "assert(abs(pi - 3.14159265358979) < 1e-10, 'pi')",
            "assert(trial == uint32(42), 'trial')",
            "assert(active == true, 'active')",
            "assert(numel(samples) == 8, 'samples length')",
            "assert(isequal(size(result), [2 3]), 'result size')",
            "assert(numel(signal) == 3, 'signal length')",
            "assert(iscomplex(signal), 'signal is complex')",
            "assert(strcmp(phase, 'Done'), 'phase')",
            "assert(isstruct(config), 'config is struct')",
            "assert(strcmp(config.tag, 'ship_it'), 'config.tag')",
            "assert(strcmp(note, 'looks good'), 'note')",
            "assert(~exist('skipped', 'var'), 'skipped should be absent')",
            "disp('experiment.mat OK')",
        ],
    );
    let v = Everything {
        name: "full_run".into(),
        pi: std::f64::consts::PI,
        trial: 42,
        active: true,
        samples: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        result: Matrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        signal: vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(-0.5, 1.0),
            Complex64::new(0.0, -1.5),
        ],
        phase: Phase::Done,
        config: Config {
            threshold: 0.99,
            tag: "ship_it".into(),
            max_iter: 5000,
        },
        note: Some("looks good".into()),
        skipped: None,
    };
    mat::to_file(&v, dir.join("experiment.mat")).unwrap();
    println!("wrote: experiment.mat");
}

// ---------------------------------------------------------------------------

fn announce(filename: &str, commands: &[&str]) {
    println!();
    println!("--- {filename} ---");
    println!("  >> load {filename}");
    for line in commands {
        println!("  >> {line}");
    }
}

// ===========================================================================
// Edge-case stress fixtures. `verify.m` has detailed assertions for these;
// no per-file `announce()` listing is needed (the verify script runs them
// all at once).
// ===========================================================================

/// Extreme numeric values: IEEE specials, integer limits, subnormals.
#[derive(Serialize, Deserialize)]
struct Extremes {
    nan64: f64,
    pos_inf: f64,
    neg_inf: f64,
    neg_zero: f64,
    subnormal: f64,
    nan32: f32,
    pos_inf32: f32,
    i64_min: i64,
    i64_max: i64,
    i32_min: i32,
    i32_max: i32,
    u64_max: u64,
    i8_min: i8,
    i8_max: i8,
    u8_max: u8,
    nan_vec: Vec<f64>,
    i64_extremes: Vec<i64>,
}

fn write_extremes(dir: &Path) {
    let v = Extremes {
        nan64: f64::NAN,
        pos_inf: f64::INFINITY,
        neg_inf: f64::NEG_INFINITY,
        neg_zero: -0.0_f64,
        subnormal: f64::from_bits(1), // smallest positive subnormal
        nan32: f32::NAN,
        pos_inf32: f32::INFINITY,
        i64_min: i64::MIN,
        i64_max: i64::MAX,
        i32_min: i32::MIN,
        i32_max: i32::MAX,
        u64_max: u64::MAX,
        i8_min: i8::MIN,
        i8_max: i8::MAX,
        u8_max: u8::MAX,
        nan_vec: vec![1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0],
        i64_extremes: vec![i64::MIN, -1, 0, 1, i64::MAX],
    };
    mat::to_file(&v, dir.join("extremes.mat")).unwrap();
    println!("wrote: extremes.mat");
}

/// Matrix orientation edges — 1x1, row vector, column vector, non-square,
/// plus a 3x3 with per-cell distinct values to catch any row/col swap.
#[derive(Serialize, Deserialize)]
struct Shapes {
    m_1x1: Matrix<f64>,
    m_1x5: Matrix<f64>,
    m_5x1: Matrix<f64>,
    m_2x3: Matrix<f64>,
    m_3x2: Matrix<f64>,
    m_3x3: Matrix<f64>,
}

fn write_shapes(dir: &Path) {
    let v = Shapes {
        m_1x1: Matrix::from_row_major(1, 1, vec![42.0]),
        m_1x5: Matrix::from_row_major(1, 5, vec![10.0, 20.0, 30.0, 40.0, 50.0]),
        m_5x1: Matrix::from_row_major(5, 1, vec![10.0, 20.0, 30.0, 40.0, 50.0]),
        m_2x3: Matrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        m_3x2: Matrix::from_row_major(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        m_3x3: Matrix::from_row_major(
            3,
            3,
            vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0],
        ),
    };
    mat::to_file(&v, dir.join("shapes.mat")).unwrap();
    println!("wrote: shapes.mat");
}

/// Every integer matrix class plus logical and single.
#[derive(Serialize, Deserialize)]
struct IntMatrices {
    m_i8: Matrix<i8>,
    m_i16: Matrix<i16>,
    m_i32: Matrix<i32>,
    m_i64: Matrix<i64>,
    m_u8: Matrix<u8>,
    m_u16: Matrix<u16>,
    m_u32: Matrix<u32>,
    m_u64: Matrix<u64>,
    m_bool: Matrix<bool>,
    m_f32: Matrix<f32>,
}

fn write_int_matrices(dir: &Path) {
    let v = IntMatrices {
        m_i8: Matrix::from_row_major(2, 2, vec![-128_i8, 127, 0, -1]),
        m_i16: Matrix::from_row_major(2, 2, vec![-32768_i16, 32767, 0, -1]),
        m_i32: Matrix::from_row_major(2, 2, vec![-1_i32, 2, 3, 4]),
        m_i64: Matrix::from_row_major(2, 2, vec![-1_i64, 2, 3, i64::MAX]),
        m_u8: Matrix::from_row_major(2, 3, vec![0_u8, 1, 2, 253, 254, 255]),
        m_u16: Matrix::from_row_major(2, 2, vec![0_u16, 1, 65534, 65535]),
        m_u32: Matrix::from_row_major(2, 2, vec![0_u32, 1, 4294967294, 4294967295]),
        m_u64: Matrix::from_row_major(2, 2, vec![0_u64, 1, 2, u64::MAX]),
        m_bool: Matrix::from_row_major(2, 2, vec![true, false, false, true]),
        m_f32: Matrix::from_row_major(2, 2, vec![1.5_f32, 2.5, 3.5, 4.5]),
    };
    mat::to_file(&v, dir.join("int_matrices.mat")).unwrap();
    println!("wrote: int_matrices.mat");
}

/// Unicode strings: Latin-1 accents, CJK, emoji surrogate pair, multi-line,
/// single char, long ASCII.
#[derive(Serialize, Deserialize)]
struct Unicode {
    latin1: String,
    cjk: String,
    emoji: String,
    mixed: String,
    multiline: String,
    one_char: String,
    long_ascii: String,
}

fn write_unicode(dir: &Path) {
    let v = Unicode {
        latin1: "café — naïve — résumé".into(),
        cjk: "日本語テスト".into(),
        emoji: "🎉".into(), // U+1F389: requires a UTF-16 surrogate pair
        mixed: "é日🎉A".into(),
        multiline: "line1\nline2\tindented\nend".into(),
        one_char: "X".into(),
        long_ascii: "abcdefghij".repeat(500),
    };
    mat::to_file(&v, dir.join("unicode.mat")).unwrap();
    println!("wrote: unicode.mat");
}

/// Complex edge cases: NaN, Inf, pure real/imaginary, f32 complex.
#[derive(Serialize, Deserialize)]
struct ComplexEdges {
    z_nan: Complex64,
    z_inf: Complex64,
    z_zero: Complex64,
    z_pure_imag: Complex64,
    z_pure_real: Complex64,
    z32: Complex32,
    z32_vec: Vec<Complex32>,
    cmat: Matrix<f64>,
}

fn write_complex_edges(dir: &Path) {
    let v = ComplexEdges {
        z_nan: Complex64::new(f64::NAN, 0.0),
        z_inf: Complex64::new(f64::INFINITY, f64::NEG_INFINITY),
        z_zero: Complex64::new(0.0, 0.0),
        z_pure_imag: Complex64::new(0.0, 2.5),
        z_pure_real: Complex64::new(3.5, 0.0),
        z32: Complex32::new(1.25, -0.5),
        z32_vec: vec![
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 1.0),
            Complex32::new(-1.0, -1.0),
        ],
        cmat: Matrix::from_row_major(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
    };
    mat::to_file(&v, dir.join("complex_edges.mat")).unwrap();
    println!("wrote: complex_edges.mat");
}

/// 4-level deep struct nesting.
#[derive(Serialize, Deserialize)]
struct Root {
    label: String,
    inner: Inner,
}
#[derive(Serialize, Deserialize)]
struct Inner {
    depth: u32,
    sub: SubInner,
}
#[derive(Serialize, Deserialize)]
struct SubInner {
    tag: String,
    leaf: Leaf,
}
#[derive(Serialize, Deserialize)]
struct Leaf {
    id: u64,
    values: Vec<f64>,
}

fn write_deep_nested(dir: &Path) {
    #[derive(Serialize)]
    struct Wrap {
        root: Root,
    }
    let v = Wrap {
        root: Root {
            label: "top".into(),
            inner: Inner {
                depth: 2,
                sub: SubInner {
                    tag: "middle".into(),
                    leaf: Leaf {
                        id: 12345,
                        values: vec![1.5, 2.5, 3.5],
                    },
                },
            },
        },
    };
    mat::to_file(&v, dir.join("deep_nested.mat")).unwrap();
    println!("wrote: deep_nested.mat");
}

/// Extended boolean cases — length-1 vec, 3x3 logical matrix, longer vec.
#[derive(Serialize, Deserialize)]
struct BoolExt {
    single: Vec<bool>,
    mat: Matrix<bool>,
    flags: Vec<bool>,
}

fn write_bool_ext(dir: &Path) {
    let v = BoolExt {
        single: vec![true],
        mat: Matrix::from_row_major(
            3,
            3,
            vec![true, false, true, false, true, false, true, false, true],
        ),
        flags: vec![true, true, false, true, false, false, true],
    };
    mat::to_file(&v, dir.join("bool_ext.mat")).unwrap();
    println!("wrote: bool_ext.mat");
}

/// Empty containers of every primitive type.
#[derive(Serialize, Deserialize)]
struct EmptyVariants {
    e_f64: Vec<f64>,
    e_f32: Vec<f32>,
    e_i32: Vec<i32>,
    e_u8: Vec<u8>,
    e_bool: Vec<bool>,
    e_str: String,
}

fn write_empty_variants(dir: &Path) {
    let v = EmptyVariants {
        e_f64: vec![],
        e_f32: vec![],
        e_i32: vec![],
        e_u8: vec![],
        e_bool: vec![],
        e_str: String::new(),
    };
    mat::to_file(&v, dir.join("empty_variants.mat")).unwrap();
    println!("wrote: empty_variants.mat");
}

/// Large matrix (100x50) with position-coded values (`value = row*1000 + col`)
/// so specific cells can be verified to rule out any transpose off-by-one at
/// size.
#[derive(Serialize, Deserialize)]
struct LargeMatrix {
    m: Matrix<f64>,
    rows: u32,
    cols: u32,
}

fn write_large_matrix(dir: &Path) {
    let rows = 100usize;
    let cols = 50usize;
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            data.push((r * 1000 + c) as f64);
        }
    }
    let v = LargeMatrix {
        m: Matrix::from_row_major(rows, cols, data),
        rows: rows as u32,
        cols: cols as u32,
    };
    mat::to_file(&v, dir.join("large_matrix.mat")).unwrap();
    println!("wrote: large_matrix.mat");
}

// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
struct Point {
    x: f64,
    y: f64,
}

#[derive(Serialize, Deserialize)]
struct CellsRoot {
    /// `Vec<Struct>` lowers to a 3x1 cell array of struct.
    points: Vec<Point>,
    /// `Vec<Option<Struct>>` with `None` interspersed: the `None` slot becomes
    /// `struct([])` (an empty struct array).
    optionals: Vec<Option<Point>>,
    /// `Vec<Vec<Option<Struct>>>`: an outer cell whose elements are themselves
    /// cells, with `None` slots becoming `struct([])` markers.
    grid: Vec<Vec<Option<Point>>>,
    /// Ragged `Vec<Vec<f64>>` falls back to a cell of doubles instead of
    /// erroring on the non-uniform inner lengths.
    ragged: Vec<Vec<f64>>,
}

fn write_cells(dir: &Path) {
    announce(
        "cells.mat",
        &[
            "assert(iscell(points), 'points iscell')",
            "assert(numel(points) == 3, 'points length')",
            "assert(points{1}.x == 1.0 && points{1}.y == 2.0, 'points{1}')",
            "assert(points{3}.x == 5.0 && points{3}.y == 6.0, 'points{3}')",
            "assert(iscell(optionals), 'optionals iscell')",
            "assert(numel(optionals) == 3, 'optionals length')",
            "assert(isstruct(optionals{2}) && isempty(fieldnames(optionals{2})), 'optionals{2} is struct([])')",
            "assert(optionals{1}.x == 10.0, 'optionals{1}')",
            "assert(optionals{3}.x == 30.0, 'optionals{3}')",
            "assert(iscell(grid), 'grid iscell')",
            "assert(numel(grid) == 2, 'grid outer length')",
            "assert(iscell(grid{1}), 'grid{1} iscell')",
            "assert(numel(grid{1}) == 2, 'grid{1} length')",
            "assert(grid{1}{1}.x == 100.0, 'grid{1}{1}.x')",
            "assert(isstruct(grid{1}{2}) && isempty(fieldnames(grid{1}{2})), 'grid{1}{2} is struct([])')",
            "assert(isstruct(grid{2}{1}) && isempty(fieldnames(grid{2}{1})), 'grid{2}{1} is struct([])')",
            "assert(grid{2}{2}.x == 200.0, 'grid{2}{2}.x')",
            "assert(iscell(ragged), 'ragged iscell')",
            "assert(numel(ragged) == 2, 'ragged length')",
            "assert(isequal(ragged{1}(:), [1; 2; 3]), 'ragged{1}')",
            "assert(isequal(ragged{2}(:), [4; 5]), 'ragged{2}')",
            "disp('cells.mat OK')",
        ],
    );
    let v = CellsRoot {
        points: vec![
            Point { x: 1.0, y: 2.0 },
            Point { x: 3.0, y: 4.0 },
            Point { x: 5.0, y: 6.0 },
        ],
        optionals: vec![
            Some(Point { x: 10.0, y: 11.0 }),
            None,
            Some(Point { x: 30.0, y: 31.0 }),
        ],
        grid: vec![
            vec![Some(Point { x: 100.0, y: 100.0 }), None],
            vec![None, Some(Point { x: 200.0, y: 200.0 })],
        ],
        ragged: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]],
    };
    mat::to_file(&v, dir.join("cells.mat")).unwrap();
    println!("wrote: cells.mat");
}
