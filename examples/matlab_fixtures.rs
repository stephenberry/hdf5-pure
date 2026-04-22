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

use hdf5_pure::mat::{self, Complex64, Matrix};
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

    println!();
    println!("All fixtures written to: {}", out.display());
    println!();
    println!("Quick check in MATLAB:");
    println!("  >> cd('matlab_fixtures')");
    println!("  >> !ls *.mat                  % list files");
    println!("  >> load scalars.mat           % try each in turn");
    println!("  >> whos                       % inspect variables");
    println!();
    println!("Per-file checks are printed above each `wrote:` line.");
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
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
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
    struct Wrap<'a> { e: &'a Experiment }
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
    let v = UnitEnum { phase: Phase::Running };
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
