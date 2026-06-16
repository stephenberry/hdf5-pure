//! The shortest path through `hdf5-pure`: build a file in memory, then read it
//! back. No filesystem and no C library are involved.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example quickstart
//! ```

use hdf5_pure::{AttrValue, File, FileBuilder};

fn main() {
    // ---- Write ----------------------------------------------------------
    let mut builder = FileBuilder::new();

    // A dataset. The shape defaults to `[len]`, so `with_shape` is optional
    // for a flat 1-D array; it is shown here for clarity.
    builder
        .create_dataset("temperature")
        .with_f64_data(&[22.5, 23.1, 21.8])
        .with_shape(&[3])
        .set_attr("unit", AttrValue::AsciiString("degC".into()));

    // An attribute on the root group.
    builder.set_attr("version", AttrValue::I64(2));

    // `finish` returns the file image as bytes; `write(path)` would instead
    // put them on disk. The in-memory form is what makes this WASM-friendly.
    let bytes = builder.finish().expect("serialize file");
    println!("wrote {} bytes", bytes.len());

    // ---- Read -----------------------------------------------------------
    let file = File::from_bytes(bytes).expect("parse file");

    let ds = file.dataset("temperature").expect("open dataset");
    println!("shape: {:?}", ds.shape().unwrap());
    println!("data:  {:?}", ds.read_f64().unwrap());
    println!("unit:  {:?}", ds.attrs().unwrap().get("unit"));

    let root_attrs = file.root().attrs().unwrap();
    println!("version: {:?}", root_attrs.get("version"));

    // The example doubles as a self-check.
    assert_eq!(ds.read_f64().unwrap(), vec![22.5, 23.1, 21.8]);
    assert_eq!(root_attrs.get("version"), Some(&AttrValue::I64(2)));
    println!("\nround-trip verified");
}
