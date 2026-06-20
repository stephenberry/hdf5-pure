//! Nested / embedded MCOS object reference decoding, validated against a real
//! MATLAB fixture.
//!
//! `tests/fixtures/mat_real/test_tables_v73.mat` is genuine real-MATLAB v7.3
//! output (BSD-3, `foreverallama/matio`; see `tests/fixtures/mat_real/NOTICE.md`).
//! Inside a `table` / `timetable`, a column or row-time that is itself an MCOS
//! object (`string`, `datetime`, `duration`, a struct, a user class) is stored
//! as an embedded object reference in the property heap. These tests assert the
//! columns decode to their real values rather than surfacing as the raw
//! `uint32` reference metadata. Expected values are transcribed from the
//! generator script `tests/data/generators/test_tables_gen.m` in that upstream
//! project, so the oracle is independent of this crate's own reader.
#![cfg(feature = "serde")]

use hdf5_pure::mat::{self, MatDatetime, MatDuration};
use serde::Deserialize;

fn fixture() -> Vec<u8> {
    std::fs::read("tests/fixtures/mat_real/test_tables_v73.mat")
        .expect("read test_tables_v73.mat fixture")
}

/// Deserialize one named variable from the fixture, ignoring the rest.
fn read<T: serde::de::DeserializeOwned>() -> T {
    mat::from_bytes(&fixture()).expect("decode fixture")
}

// A `table`'s opaque properties surface directly: `data` is a cell of columns,
// `varnames` is the variable-name list, `nrows`/`nvars` the shape.

#[test]
fn table_numeric_columns_still_decode() {
    // `table([1.1;2.2;3.3], [4.4;5.5;6.6])` — pure-numeric columns were already
    // readable; this guards against the embedded-ref path regressing them.
    #[derive(Deserialize)]
    struct File {
        table_numeric: Table,
    }
    #[derive(Deserialize)]
    struct Table {
        varnames: Vec<String>,
        nrows: f64,
        data: Vec<Vec<f64>>,
    }
    let f: File = read();
    assert_eq!(f.table_numeric.varnames, ["Var1", "Var2"]);
    assert_eq!(f.table_numeric.nrows, 3.0);
    assert_eq!(
        f.table_numeric.data,
        vec![vec![1.1, 2.2, 3.3], vec![4.4, 5.5, 6.6]]
    );
}

#[test]
fn table_string_column_decodes() {
    // `table(["apple";"banana";"cherry"])` — the column is a nested `string`
    // object reached by embedded reference inside the `data` cell.
    #[derive(Deserialize)]
    struct File {
        table_strings: Table,
    }
    #[derive(Deserialize)]
    struct Table {
        varnames: Vec<String>,
        data: Vec<Vec<String>>,
    }
    let f: File = read();
    assert_eq!(f.table_strings.varnames, ["Var1"]);
    assert_eq!(
        f.table_strings.data,
        vec![vec!["apple", "banana", "cherry"]]
    );
}

#[test]
fn table_datetime_and_duration_columns_decode() {
    // `table(datetime(2020,1,1)+days(0:2)', seconds([30;60;90]))` — one
    // `datetime` and one `duration` column, both nested MCOS objects.
    #[derive(Deserialize)]
    struct File {
        table_time: Table,
    }
    #[derive(Deserialize)]
    struct Table {
        varnames: Vec<String>,
        data: (MatDatetime, MatDuration),
    }
    let f: File = read();
    assert_eq!(f.table_time.varnames, ["Time", "Duration"]);
    let (time, dur) = f.table_time.data;
    // 2020-01-01, 2020-01-02, 2020-01-03 UTC in epoch milliseconds.
    assert_eq!(
        time.millis_utc,
        vec![
            1_577_836_800_000.0,
            1_577_923_200_000.0,
            1_578_009_600_000.0
        ]
    );
    // seconds(30/60/90) in milliseconds.
    assert_eq!(dur.millis, vec![30_000.0, 60_000.0, 90_000.0]);
    assert_eq!(dur.seconds(), vec![30.0, 60.0, 90.0]);
}

#[test]
fn table_nan_and_missing_string_columns_decode() {
    // `table([1.1;NaN;3.3], ["A";"";"C"])` — a numeric column with a NaN and a
    // string column with an empty element.
    #[derive(Deserialize)]
    struct File {
        table_nan: Table,
    }
    #[derive(Deserialize)]
    struct Table {
        data: (Vec<f64>, Vec<String>),
    }
    let f: File = read();
    let (nums, strs) = f.table_nan.data;
    assert_eq!(nums[0], 1.1);
    assert!(nums[1].is_nan());
    assert_eq!(nums[2], 3.3);
    assert_eq!(strs, vec!["A", "", "C"]);
}

#[test]
fn table_heterogeneous_cell_column_decodes() {
    // `table({1; 'text'; datetime(2023,1,1)})` — a single cell column mixing a
    // double, a char row, and a nested `datetime`.
    #[derive(Deserialize)]
    struct File {
        table_from_cell: Table,
    }
    #[derive(Deserialize)]
    struct Table {
        data: ((f64, String, MatDatetime),),
    }
    let f: File = read();
    let ((num, text, dt),) = f.table_from_cell.data;
    assert_eq!(num, 1.0);
    assert_eq!(text, "text");
    // datetime(2023,1,1) UTC in epoch milliseconds.
    assert_eq!(dt.millis_utc, vec![1_672_531_200_000.0]);
}

#[test]
fn table_struct_and_user_object_columns_decode() {
    // `table({S;S;S}, {obj;obj;obj})` — a struct column and a user-class
    // (`TestClasses.BasicClass`) column; both decode losslessly as structs.
    #[derive(Deserialize)]
    struct File {
        table_with_objects: Table,
    }
    #[derive(Deserialize)]
    struct Table {
        data: (Vec<SCell>, Vec<BasicClass>),
    }
    #[derive(Deserialize)]
    struct SCell {
        field1: f64,
        field2: String,
    }
    #[derive(Deserialize)]
    struct BasicClass {
        a: f64,
    }
    let f: File = read();
    let (structs, objects) = f.table_with_objects.data;
    assert_eq!(structs.len(), 3);
    assert_eq!(structs[0].field1, 123.0);
    assert_eq!(structs[0].field2, "abc");
    assert_eq!(objects.len(), 3);
    assert_eq!(objects[0].a, 1.0);
}

#[test]
fn timetable_datetime_rowtimes_decode() {
    // `timetable(datetime(2023,1,1)+days(0:2)', [1;2;3])` — the row times are a
    // nested `datetime` under the timetable's `any` wrapper.
    #[derive(Deserialize)]
    struct File {
        timetable_datetime: Timetable,
    }
    #[derive(Deserialize)]
    struct Timetable {
        any: Inner,
    }
    #[derive(Deserialize)]
    struct Inner {
        #[serde(rename = "varNames")]
        var_names: Vec<String>,
        #[serde(rename = "numRows")]
        num_rows: f64,
        #[serde(rename = "rowTimes")]
        row_times: MatDatetime,
        data: Vec<Vec<f64>>,
    }
    let f: File = read();
    let tt = f.timetable_datetime.any;
    assert_eq!(tt.var_names, ["data1"]);
    assert_eq!(tt.num_rows, 3.0);
    assert_eq!(
        tt.row_times.millis_utc,
        vec![
            1_672_531_200_000.0,
            1_672_617_600_000.0,
            1_672_704_000_000.0
        ]
    );
    assert_eq!(tt.data, vec![vec![1.0, 2.0, 3.0]]);
}

#[test]
fn timetable_duration_rowtimes_decode() {
    // `timetable(seconds([10,20,30])', [1;2;3])` — duration row times.
    #[derive(Deserialize)]
    struct File {
        timetable_duration: Timetable,
    }
    #[derive(Deserialize)]
    struct Timetable {
        any: Inner,
    }
    #[derive(Deserialize)]
    struct Inner {
        #[serde(rename = "rowTimes")]
        row_times: MatDuration,
    }
    let f: File = read();
    assert_eq!(
        f.timetable_duration.any.row_times.millis,
        vec![10_000.0, 20_000.0, 30_000.0]
    );
}
