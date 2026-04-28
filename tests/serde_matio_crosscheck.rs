#![cfg(all(feature = "matio-crosscheck", target_family = "unix"))]
//! Crosscheck the serde layer against the reference `matio` C library.
//!
//! `matio-rs` on crates.io builds its own bundled libmatio with `MAT73=OFF`,
//! so it can't read our HDF5-based v7.3 files. This test instead links
//! directly against the system libmatio (which does have HDF5/v7.3 support)
//! through a hand-written FFI.
//!
//! Enable with `--features matio-crosscheck`. Requires libmatio installed
//! (`brew install libmatio` on macOS, `apt install libmatio-dev` on Debian).

// =======================================================================
// FFI bindings to system libmatio.
// =======================================================================

#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code
)]
mod ffi {
    use std::ffi::{c_char, c_int, c_uint, c_void};

    #[repr(C)]
    pub struct mat_t {
        _private: [u8; 0],
    }

    /// Layout of `matvar_t` from matio.h. Fields are stable public API.
    #[repr(C)]
    pub struct matvar_t {
        pub nbytes: usize,
        pub rank: c_int,
        pub data_type: c_int,
        pub data_size: c_int,
        pub class_type: c_int,
        pub is_complex: c_int,
        pub is_global: c_int,
        pub is_logical: c_int,
        pub dims: *mut usize,
        pub name: *mut c_char,
        pub data: *mut c_void,
        pub mem_conserve: c_int,
        pub compression: c_int,
        pub internal: *mut c_void,
    }

    // File versions
    pub const MAT_FT_MAT73: c_uint = 0x0200;

    // Access modes
    pub const MAT_ACC_RDONLY: c_int = 0;

    // matio_classes
    pub const MAT_C_CELL: c_int = 1;
    pub const MAT_C_STRUCT: c_int = 2;
    pub const MAT_C_CHAR: c_int = 4;
    pub const MAT_C_DOUBLE: c_int = 6;
    pub const MAT_C_SINGLE: c_int = 7;
    pub const MAT_C_INT8: c_int = 8;
    pub const MAT_C_UINT8: c_int = 9;
    pub const MAT_C_INT16: c_int = 10;
    pub const MAT_C_UINT16: c_int = 11;
    pub const MAT_C_INT32: c_int = 12;
    pub const MAT_C_UINT32: c_int = 13;
    pub const MAT_C_INT64: c_int = 14;
    pub const MAT_C_UINT64: c_int = 15;

    // matio_types
    pub const MAT_T_INT8: c_int = 1;
    pub const MAT_T_UINT8: c_int = 2;
    pub const MAT_T_INT16: c_int = 3;
    pub const MAT_T_UINT16: c_int = 4;
    pub const MAT_T_INT32: c_int = 5;
    pub const MAT_T_UINT32: c_int = 6;
    pub const MAT_T_SINGLE: c_int = 7;
    pub const MAT_T_DOUBLE: c_int = 9;
    pub const MAT_T_INT64: c_int = 12;
    pub const MAT_T_UINT64: c_int = 13;

    pub const MAT_COMPRESSION_NONE: c_int = 0;

    unsafe extern "C" {
        pub unsafe fn Mat_CreateVer(
            matname: *const c_char,
            hdr_str: *const c_char,
            version: c_uint,
        ) -> *mut mat_t;
        pub unsafe fn Mat_Open(matname: *const c_char, mode: c_int) -> *mut mat_t;
        pub unsafe fn Mat_Close(mat: *mut mat_t) -> c_int;

        pub unsafe fn Mat_VarRead(mat: *mut mat_t, name: *const c_char) -> *mut matvar_t;
        pub unsafe fn Mat_VarFree(matvar: *mut matvar_t);

        pub unsafe fn Mat_VarCreate(
            name: *const c_char,
            class_type: c_int,
            data_type: c_int,
            rank: c_int,
            dims: *const usize,
            data: *const c_void,
            opt: c_int,
        ) -> *mut matvar_t;
        pub unsafe fn Mat_VarWrite(
            mat: *mut mat_t,
            matvar: *mut matvar_t,
            compress: c_int,
        ) -> c_int;

        pub unsafe fn Mat_VarGetStructFieldByName(
            matvar: *const matvar_t,
            field_name: *const c_char,
            index: usize,
        ) -> *mut matvar_t;
        pub unsafe fn Mat_VarAddStructField(
            matvar: *mut matvar_t,
            fieldname: *const c_char,
        ) -> c_int;
        pub unsafe fn Mat_VarCreateStruct2(
            name: *const c_char,
            rank: c_int,
            dims: *const usize,
            fields: *const *const c_char,
        ) -> *mut matvar_t;
        pub unsafe fn Mat_VarSetStructFieldByName(
            matvar: *mut matvar_t,
            field_name: *const c_char,
            index: usize,
            field: *mut matvar_t,
        ) -> *mut matvar_t;
    }
}

// =======================================================================
// Safe-ish wrappers. Test-quality: unwraps on error, does not impose any
// lifetime discipline on borrowed struct-field pointers.
// =======================================================================

use std::ffi::{CStr, CString, c_void};
use std::path::Path;
use std::ptr;

pub struct MatFile {
    ptr: *mut ffi::mat_t,
}

impl MatFile {
    /// Create a new empty v7.3 `.mat` file.
    pub fn create_v73(path: &Path) -> Self {
        let c = CString::new(path.to_str().unwrap()).unwrap();
        let ptr = unsafe { ffi::Mat_CreateVer(c.as_ptr(), ptr::null(), ffi::MAT_FT_MAT73) };
        assert!(!ptr.is_null(), "Mat_CreateVer failed for {path:?}");
        Self { ptr }
    }

    /// Open an existing `.mat` file for reading.
    pub fn open(path: &Path) -> Self {
        let c = CString::new(path.to_str().unwrap()).unwrap();
        let ptr = unsafe { ffi::Mat_Open(c.as_ptr(), ffi::MAT_ACC_RDONLY) };
        assert!(!ptr.is_null(), "Mat_Open failed for {path:?}");
        Self { ptr }
    }

    /// Read a named variable (fully, including data). Returns `None` if not found.
    pub fn read(&self, name: &str) -> Option<MatVar> {
        let c = CString::new(name).unwrap();
        let p = unsafe { ffi::Mat_VarRead(self.ptr, c.as_ptr()) };
        if p.is_null() {
            None
        } else {
            Some(MatVar {
                ptr: p,
                owned: true,
            })
        }
    }

    /// Write a prepared MatVar to this file.
    pub fn write(&self, var: MatVar) {
        let rc = unsafe { ffi::Mat_VarWrite(self.ptr, var.ptr, ffi::MAT_COMPRESSION_NONE) };
        assert_eq!(rc, 0, "Mat_VarWrite failed ({rc})");
        // matio keeps a reference after Mat_VarWrite; still must free on our side.
        drop(var);
    }
}

impl Drop for MatFile {
    fn drop(&mut self) {
        unsafe { ffi::Mat_Close(self.ptr) };
    }
}

/// Reference to a MATLAB variable. `owned = true` means the pointer came from
/// `Mat_VarRead` / `Mat_VarCreate` and we must free it; `owned = false` means
/// it's a child pointer (struct field) owned by a parent.
pub struct MatVar {
    ptr: *mut ffi::matvar_t,
    owned: bool,
}

impl MatVar {
    // ---- builders ----

    fn create_numeric<T: Copy>(
        name: &str,
        class_type: i32,
        data_type: i32,
        dims: &[usize],
        data: &[T],
    ) -> Self {
        let c = CString::new(name).unwrap();
        let ptr = unsafe {
            ffi::Mat_VarCreate(
                c.as_ptr(),
                class_type,
                data_type,
                dims.len() as i32,
                dims.as_ptr(),
                data.as_ptr() as *const c_void,
                0,
            )
        };
        assert!(!ptr.is_null(), "Mat_VarCreate({name}) failed");
        Self { ptr, owned: true }
    }

    pub fn f64_scalar(name: &str, v: f64) -> Self {
        Self::create_numeric(name, ffi::MAT_C_DOUBLE, ffi::MAT_T_DOUBLE, &[1, 1], &[v])
    }
    pub fn f64_row_vec(name: &str, v: &[f64]) -> Self {
        Self::create_numeric(name, ffi::MAT_C_DOUBLE, ffi::MAT_T_DOUBLE, &[1, v.len()], v)
    }
    /// Build a 2-D f64 matrix. `data` must be in column-major order (MATLAB).
    pub fn f64_matrix(name: &str, rows: usize, cols: usize, col_major: &[f64]) -> Self {
        assert_eq!(col_major.len(), rows * cols);
        Self::create_numeric(
            name,
            ffi::MAT_C_DOUBLE,
            ffi::MAT_T_DOUBLE,
            &[rows, cols],
            col_major,
        )
    }
    pub fn i32_scalar(name: &str, v: i32) -> Self {
        Self::create_numeric(name, ffi::MAT_C_INT32, ffi::MAT_T_INT32, &[1, 1], &[v])
    }
    pub fn i32_row_vec(name: &str, v: &[i32]) -> Self {
        Self::create_numeric(name, ffi::MAT_C_INT32, ffi::MAT_T_INT32, &[1, v.len()], v)
    }
    pub fn u32_scalar(name: &str, v: u32) -> Self {
        Self::create_numeric(name, ffi::MAT_C_UINT32, ffi::MAT_T_UINT32, &[1, 1], &[v])
    }

    // ---- accessors ----

    pub fn class_type(&self) -> i32 {
        unsafe { (*self.ptr).class_type }
    }
    pub fn rank(&self) -> i32 {
        unsafe { (*self.ptr).rank }
    }
    pub fn dims(&self) -> Vec<usize> {
        let r = self.rank() as usize;
        let dp = unsafe { (*self.ptr).dims };
        unsafe { std::slice::from_raw_parts(dp, r) }.to_vec()
    }
    pub fn nelements(&self) -> usize {
        self.dims().iter().product()
    }
    pub fn is_complex(&self) -> bool {
        unsafe { (*self.ptr).is_complex != 0 }
    }
    pub fn name(&self) -> String {
        let p = unsafe { (*self.ptr).name };
        if p.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }

    /// Read data as a `Vec<T>`. The caller is responsible for picking a T
    /// matching the on-disk class.
    pub fn data_as<T: Copy>(&self) -> Vec<T> {
        let n = self.nelements();
        let p = unsafe { (*self.ptr).data } as *const T;
        unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
    }

    pub fn scalar_f64(&self) -> f64 {
        assert_eq!(self.class_type(), ffi::MAT_C_DOUBLE);
        assert_eq!(self.nelements(), 1);
        self.data_as::<f64>()[0]
    }
    pub fn scalar_i32(&self) -> i32 {
        assert_eq!(self.class_type(), ffi::MAT_C_INT32);
        assert_eq!(self.nelements(), 1);
        self.data_as::<i32>()[0]
    }
    pub fn scalar_u32(&self) -> u32 {
        assert_eq!(self.class_type(), ffi::MAT_C_UINT32);
        assert_eq!(self.nelements(), 1);
        self.data_as::<u32>()[0]
    }

    /// Access a struct field by name. The returned borrow is invalidated if
    /// this parent is dropped; tests must not move the parent.
    pub fn field(&self, name: &str) -> Option<MatVar> {
        let c = CString::new(name).unwrap();
        let p = unsafe { ffi::Mat_VarGetStructFieldByName(self.ptr, c.as_ptr(), 0) };
        if p.is_null() {
            None
        } else {
            Some(MatVar {
                ptr: p,
                owned: false,
            })
        }
    }

    // ---- struct building (for write tests) ----

    /// Build an empty struct variable with the given field names. Fields are
    /// populated afterwards via [`set_field`](Self::set_field).
    pub fn empty_struct(name: &str, field_names: &[&str]) -> Self {
        let c = CString::new(name).unwrap();
        let dims = [1usize, 1];
        // Mat_VarCreateStruct2 expects a NULL-terminated array of C strings
        // (unlike the deprecated Mat_VarCreateStruct which takes a count).
        let cfields: Vec<CString> = field_names
            .iter()
            .map(|f| CString::new(*f).unwrap())
            .collect();
        let mut cfield_ptrs: Vec<*const std::ffi::c_char> =
            cfields.iter().map(|c| c.as_ptr()).collect();
        cfield_ptrs.push(std::ptr::null());
        let ptr = unsafe {
            ffi::Mat_VarCreateStruct2(c.as_ptr(), 2, dims.as_ptr(), cfield_ptrs.as_ptr())
        };
        assert!(!ptr.is_null(), "Mat_VarCreateStruct2({name}) failed");
        MatVar { ptr, owned: true }
    }

    /// Attach `value` as the named field. Transfers ownership of `value`.
    pub fn set_field(&mut self, name: &str, value: MatVar) {
        let c = CString::new(name).unwrap();
        let raw = value.ptr;
        // After transfer, the struct owns the child. Prevent our Drop from
        // double-freeing.
        std::mem::forget(value);
        unsafe {
            ffi::Mat_VarSetStructFieldByName(self.ptr, c.as_ptr(), 0, raw);
        }
    }
}

impl Drop for MatVar {
    fn drop(&mut self) {
        if self.owned {
            unsafe { ffi::Mat_VarFree(self.ptr) }
        }
    }
}

// =======================================================================
// Tests
// =======================================================================

use hdf5_pure::mat::{self, Matrix};
use serde::{Deserialize, Serialize};
use std::sync::{Mutex, MutexGuard};
use tempfile::tempdir;

/// Serializes all libmatio calls — HDF5 (which libmatio calls internally)
/// isn't thread-safe by default and cargo runs tests in parallel.
static MATIO_LOCK: Mutex<()> = Mutex::new(());
fn matio_lock() -> MutexGuard<'static, ()> {
    // Intentionally ignore poisoning — a panicking test should not break
    // later ones.
    MATIO_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Scalars {
    x: f64,
    n: i32,
    u: u32,
}

#[test]
fn matio_reads_scalars_from_hdf5_pure() {
    let _g = matio_lock();
    let dir = tempdir().unwrap();
    let path = dir.path().join("scalars.mat");
    mat::to_file(
        &Scalars {
            x: 1.5,
            n: -7,
            u: 42,
        },
        &path,
    )
    .unwrap();

    let f = MatFile::open(&path);
    assert_eq!(f.read("x").unwrap().scalar_f64(), 1.5);
    assert_eq!(f.read("n").unwrap().scalar_i32(), -7);
    assert_eq!(f.read("u").unwrap().scalar_u32(), 42);
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Vectors {
    xs: Vec<f64>,
    ns: Vec<i32>,
}

#[test]
fn matio_reads_vectors_from_hdf5_pure() {
    let _g = matio_lock();
    let dir = tempdir().unwrap();
    let path = dir.path().join("vectors.mat");
    mat::to_file(
        &Vectors {
            xs: vec![1.0, 2.0, 3.0, 4.0],
            ns: vec![-1, 0, 1],
        },
        &path,
    )
    .unwrap();

    // matio reports dims in MATLAB order (post-transpose), so our HDF5 shape
    // [1, N] — which represents a MATLAB column vector — is presented as
    // [N, 1]. The values are unchanged.
    let f = MatFile::open(&path);
    let xs_v = f.read("xs").unwrap();
    assert_eq!(xs_v.class_type(), ffi::MAT_C_DOUBLE);
    assert_eq!(xs_v.dims(), vec![4, 1]);
    assert_eq!(xs_v.data_as::<f64>(), vec![1.0, 2.0, 3.0, 4.0]);

    let ns_v = f.read("ns").unwrap();
    assert_eq!(ns_v.class_type(), ffi::MAT_C_INT32);
    assert_eq!(ns_v.dims(), vec![3, 1]);
    assert_eq!(ns_v.data_as::<i32>(), vec![-1, 0, 1]);
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct MatrixOnly {
    m: Matrix<f64>,
}

#[test]
fn matio_reads_2d_matrix_from_hdf5_pure() {
    let _g = matio_lock();
    let dir = tempdir().unwrap();
    let path = dir.path().join("matrix.mat");

    // Rust row-major 2×3:
    //   [[1 2 3]
    //    [4 5 6]]
    let v = MatrixOnly {
        m: Matrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    };
    mat::to_file(&v, &path).unwrap();

    let f = MatFile::open(&path);
    let m = f.read("m").unwrap();
    assert_eq!(m.class_type(), ffi::MAT_C_DOUBLE);
    // matio exposes the MATLAB shape directly: 2 rows × 3 cols.
    assert_eq!(m.dims(), vec![2, 3]);
    // Stored column-major.
    assert_eq!(m.data_as::<f64>(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Config {
    threshold: f64,
    trial: i32,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Nested {
    name: i32, // using i32 to keep matio-side decoding simple
    config: Config,
}

#[test]
fn matio_reads_nested_struct_from_hdf5_pure() {
    let _g = matio_lock();
    let dir = tempdir().unwrap();
    let path = dir.path().join("nested.mat");

    mat::to_file(
        &Nested {
            name: 99,
            config: Config {
                threshold: 0.25,
                trial: 3,
            },
        },
        &path,
    )
    .unwrap();

    let f = MatFile::open(&path);
    assert_eq!(f.read("name").unwrap().scalar_i32(), 99);

    let cfg = f.read("config").unwrap();
    assert_eq!(cfg.class_type(), ffi::MAT_C_STRUCT);
    let threshold = cfg.field("threshold").unwrap();
    assert_eq!(threshold.scalar_f64(), 0.25);
    let trial = cfg.field("trial").unwrap();
    assert_eq!(trial.scalar_i32(), 3);
}

// ---------------------------------------------------------------------------
// matio → hdf5-pure
// ---------------------------------------------------------------------------

#[test]
fn hdf5_pure_reads_scalars_written_by_matio() {
    let _g = matio_lock();
    let dir = tempdir().unwrap();
    let path = dir.path().join("matio_scalars.mat");
    {
        let f = MatFile::create_v73(&path);
        f.write(MatVar::f64_scalar("x", 1.5));
        f.write(MatVar::i32_scalar("n", -7));
        f.write(MatVar::u32_scalar("u", 42));
    }

    let parsed: Scalars = mat::from_file(&path).unwrap();
    assert_eq!(
        parsed,
        Scalars {
            x: 1.5,
            n: -7,
            u: 42
        }
    );
}

#[test]
fn hdf5_pure_reads_vectors_written_by_matio() {
    let _g = matio_lock();
    let dir = tempdir().unwrap();
    let path = dir.path().join("matio_vectors.mat");
    {
        let f = MatFile::create_v73(&path);
        f.write(MatVar::f64_row_vec("xs", &[1.0, 2.0, 3.0, 4.0]));
        f.write(MatVar::i32_row_vec("ns", &[-1, 0, 1]));
    }

    let parsed: Vectors = mat::from_file(&path).unwrap();
    assert_eq!(
        parsed,
        Vectors {
            xs: vec![1.0, 2.0, 3.0, 4.0],
            ns: vec![-1, 0, 1]
        }
    );
}

#[test]
fn hdf5_pure_reads_matrix_written_by_matio() {
    let _g = matio_lock();
    let dir = tempdir().unwrap();
    let path = dir.path().join("matio_matrix.mat");
    {
        let f = MatFile::create_v73(&path);
        // MATLAB 2×3 matrix in column-major order:
        //   [[1 2 3]
        //    [4 5 6]]  -> columns [1,4,2,5,3,6]
        f.write(MatVar::f64_matrix(
            "m",
            2,
            3,
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        ));
    }

    let parsed: MatrixOnly = mat::from_file(&path).unwrap();
    assert_eq!(parsed.m.rows(), 2);
    assert_eq!(parsed.m.cols(), 3);
    assert_eq!(parsed.m.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn hdf5_pure_reads_nested_struct_written_by_matio() {
    let _g = matio_lock();
    let dir = tempdir().unwrap();
    let path = dir.path().join("matio_nested.mat");
    {
        let f = MatFile::create_v73(&path);
        f.write(MatVar::i32_scalar("name", 99));

        let mut cfg = MatVar::empty_struct("config", &["threshold", "trial"]);
        cfg.set_field("threshold", MatVar::f64_scalar("threshold", 0.25));
        cfg.set_field("trial", MatVar::i32_scalar("trial", 3));
        f.write(cfg);
    }

    let parsed: Nested = mat::from_file(&path).unwrap();
    assert_eq!(
        parsed,
        Nested {
            name: 99,
            config: Config {
                threshold: 0.25,
                trial: 3
            }
        }
    );
}

// ---------------------------------------------------------------------------
// Full roundtrip: hdf5-pure → matio → hdf5-pure
// ---------------------------------------------------------------------------

#[test]
fn full_roundtrip_via_matio() {
    let _g = matio_lock();
    let dir = tempdir().unwrap();
    let path1 = dir.path().join("a.mat");
    let path2 = dir.path().join("b.mat");

    mat::to_file(
        &Vectors {
            xs: vec![0.5, 1.5, 2.5],
            ns: vec![10, 20, 30],
        },
        &path1,
    )
    .unwrap();

    // matio reads → matio rewrites
    {
        let src = MatFile::open(&path1);
        let xs = src.read("xs").unwrap().data_as::<f64>();
        let ns = src.read("ns").unwrap().data_as::<i32>();

        let dst = MatFile::create_v73(&path2);
        dst.write(MatVar::f64_row_vec("xs", &xs));
        dst.write(MatVar::i32_row_vec("ns", &ns));
    }

    let parsed: Vectors = mat::from_file(&path2).unwrap();
    assert_eq!(parsed.xs, vec![0.5, 1.5, 2.5]);
    assert_eq!(parsed.ns, vec![10, 20, 30]);
}

/// Regression: a column-vector matrix written by matio (MATLAB [N,1]) must
/// round-trip through our Matrix<T> reader as a Matrix of shape (N, 1), not
/// (1, N). This exercises the reader path that used to silently flatten 2-D
/// datasets with a unit dimension into Vec1D and lose the orientation.
#[test]
fn hdf5_pure_reads_column_vector_matrix_from_matio() {
    use hdf5_pure::mat::Matrix;
    let _g = matio_lock();
    let dir = tempdir().unwrap();
    let path = dir.path().join("colvec.mat");
    {
        let f = MatFile::create_v73(&path);
        // MATLAB 3×1 column. Column-major bytes of a 3×1 are just [a, b, c].
        f.write(MatVar::f64_matrix("m", 3, 1, &[10.0, 20.0, 30.0]));
    }

    #[derive(serde::Deserialize)]
    struct ColOnly {
        m: Matrix<f64>,
    }
    let parsed: ColOnly = mat::from_file(&path).unwrap();
    assert_eq!(parsed.m.rows(), 3);
    assert_eq!(parsed.m.cols(), 1);
    assert_eq!(parsed.m.data(), &[10.0, 20.0, 30.0]);
}

/// Companion: row-vector matrix from matio round-trips as (1, N).
#[test]
fn hdf5_pure_reads_row_vector_matrix_from_matio() {
    use hdf5_pure::mat::Matrix;
    let _g = matio_lock();
    let dir = tempdir().unwrap();
    let path = dir.path().join("rowvec.mat");
    {
        let f = MatFile::create_v73(&path);
        // MATLAB 1×4 row. Column-major bytes of a 1×4 are [a, b, c, d].
        f.write(MatVar::f64_matrix("m", 1, 4, &[10.0, 20.0, 30.0, 40.0]));
    }

    #[derive(serde::Deserialize)]
    struct RowOnly {
        m: Matrix<f64>,
    }
    let parsed: RowOnly = mat::from_file(&path).unwrap();
    assert_eq!(parsed.m.rows(), 1);
    assert_eq!(parsed.m.cols(), 4);
    assert_eq!(parsed.m.data(), &[10.0, 20.0, 30.0, 40.0]);
}
