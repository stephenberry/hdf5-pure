use hdf5_pure::{AttrValue, CompoundTypeBuilder, DType, File, FileBuilder, make_f64_type};

#[test]
fn roundtrip_f64_dataset() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0])
        .with_shape(&[4]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![4]);
    assert_eq!(ds.dtype().unwrap(), DType::F64);
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn roundtrip_i32_dataset() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("values")
        .with_i32_data(&[10, 20, 30]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("values").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::I32);
    assert_eq!(ds.read_i32().unwrap(), vec![10, 20, 30]);
}

#[test]
fn roundtrip_u8_dataset() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("bytes")
        .with_u8_data(&[0xFF, 0x00, 0xAB]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("bytes").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::U8);
}

#[test]
fn roundtrip_with_attributes() {
    let mut builder = FileBuilder::new();
    builder.set_attr("version", AttrValue::I64(2));
    builder
        .create_dataset("data")
        .with_f64_data(&[42.0])
        .set_attr("unit", AttrValue::String("m/s".into()));
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();

    let root_attrs = file.root().attrs().unwrap();
    assert_eq!(root_attrs.get("version"), Some(&AttrValue::I64(2)));

    let ds = file.dataset("data").unwrap();
    let ds_attrs = ds.attrs().unwrap();
    assert_eq!(
        ds_attrs.get("unit"),
        Some(&AttrValue::String("m/s".into()))
    );
}

#[test]
fn roundtrip_group_with_dataset() {
    let mut builder = FileBuilder::new();
    let mut grp = builder.create_group("sensors");
    grp.create_dataset("temperature")
        .with_f64_data(&[20.5, 21.0, 22.3]);
    grp.set_attr("location", AttrValue::String("lab".into()));
    let finished = grp.finish();
    builder.add_group(finished);

    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let grp = file.group("sensors").unwrap();
    let datasets = grp.datasets().unwrap();
    assert_eq!(datasets, vec!["temperature"]);

    let ds = file.dataset("sensors/temperature").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![20.5, 21.0, 22.3]);

    let grp_attrs = grp.attrs().unwrap();
    assert_eq!(
        grp_attrs.get("location"),
        Some(&AttrValue::String("lab".into()))
    );
}

#[test]
fn roundtrip_multiple_datasets() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("x")
        .with_f64_data(&[1.0, 2.0, 3.0]);
    builder
        .create_dataset("y")
        .with_f64_data(&[4.0, 5.0, 6.0]);
    builder
        .create_dataset("z")
        .with_i32_data(&[7, 8, 9]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(file.dataset("x").unwrap().read_f64().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(file.dataset("y").unwrap().read_f64().unwrap(), vec![4.0, 5.0, 6.0]);
    assert_eq!(file.dataset("z").unwrap().read_i32().unwrap(), vec![7, 8, 9]);
}

#[test]
fn roundtrip_write_to_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&[1.0, 2.0]);
    builder.write(&path).unwrap();

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("data").unwrap().read_f64().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn roundtrip_i8_dataset() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i8_data(&[-128, -1, 0, 1, 127]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::I8);
    assert_eq!(ds.read_i8().unwrap(), vec![-128, -1, 0, 1, 127]);
}

#[test]
fn roundtrip_i16_dataset() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i16_data(&[-32768, -1, 0, 1, 32767]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::I16);
    assert_eq!(ds.read_i16().unwrap(), vec![-32768, -1, 0, 1, 32767]);
}

#[test]
fn roundtrip_u16_dataset() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_u16_data(&[0, 1, 1000, 65535]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::U16);
    assert_eq!(ds.read_u16().unwrap(), vec![0, 1, 1000, 65535]);
}

#[test]
fn roundtrip_u32_dataset() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_u32_data(&[0, 1, 1_000_000, u32::MAX]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::U32);
    assert_eq!(ds.read_u32().unwrap(), vec![0, 1, 1_000_000, u32::MAX]);
}

#[test]
fn roundtrip_u64_dataset() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_u64_data(&[0, 1, u64::MAX]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::U64);
    assert_eq!(ds.read_u64().unwrap(), vec![0, 1, u64::MAX]);
}

#[test]
fn roundtrip_compound_complex64() {
    let complex_type = CompoundTypeBuilder::new()
        .f64_field("real")
        .f64_field("imag")
        .build();

    let values: Vec<(f64, f64)> = vec![(1.0, 2.0), (3.0, 4.0)];
    let mut raw = Vec::new();
    for (r, i) in &values {
        raw.extend_from_slice(&r.to_le_bytes());
        raw.extend_from_slice(&i.to_le_bytes());
    }

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("complex")
        .with_compound_data(complex_type, raw, 2);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("complex").unwrap();
    match ds.dtype().unwrap() {
        DType::Compound(fields) => {
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].0, "real");
            assert_eq!(fields[1].0, "imag");
        }
        other => panic!("expected compound, got {other}"),
    }
}

#[test]
fn roundtrip_userblock_512() {
    let mut builder = FileBuilder::new();
    builder.with_userblock(512);
    builder.create_dataset("data").with_f64_data(&[1.0, 2.0, 3.0]);
    let mut bytes = builder.finish().unwrap();

    // Verify userblock is at the start (512 zero bytes)
    assert!(bytes.len() > 512);
    assert!(bytes[..512].iter().all(|&b| b == 0));

    // HDF5 signature should start at offset 512
    assert_eq!(&bytes[512..520], b"\x89HDF\r\n\x1a\n");

    // Write some userblock data (e.g., MATLAB header)
    bytes[0..6].copy_from_slice(b"MATLAB");

    // File should still be readable
    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn roundtrip_nested_groups() {
    let mut builder = FileBuilder::new();

    let mut outer = builder.create_group("outer");
    outer.create_dataset("outer_data").with_f64_data(&[1.0]);

    let mut inner = outer.create_group("inner");
    inner.create_dataset("inner_data").with_f64_data(&[2.0, 3.0]);
    inner.set_attr("depth", AttrValue::I64(2));
    outer.add_group(inner.finish());

    builder.add_group(outer.finish());

    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();

    // Read outer dataset
    let ds = file.dataset("outer/outer_data").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![1.0]);

    // Read inner dataset
    let ds = file.dataset("outer/inner/inner_data").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![2.0, 3.0]);

    // Read inner group attrs
    let inner_grp = file.group("outer/inner").unwrap();
    let attrs = inner_grp.attrs().unwrap();
    assert_eq!(attrs.get("depth"), Some(&AttrValue::I64(2)));

    // Verify group hierarchy
    let outer_grp = file.group("outer").unwrap();
    let sub_groups = outer_grp.groups().unwrap();
    assert_eq!(sub_groups, vec!["inner"]);
    let datasets = outer_grp.datasets().unwrap();
    assert_eq!(datasets, vec!["outer_data"]);
}

#[test]
fn roundtrip_i32_u32_attributes() {
    let mut builder = FileBuilder::new();
    builder.set_attr("MATLAB_int_decode", AttrValue::I32(1));
    builder.set_attr("MATLAB_empty", AttrValue::U32(1));
    builder.create_dataset("x").with_f64_data(&[0.0]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let attrs = file.root().attrs().unwrap();

    // I32 is read back as I64 (the reader promotes)
    assert_eq!(attrs.get("MATLAB_int_decode"), Some(&AttrValue::I64(1)));
    // U32 is read back as U64 (the reader promotes)
    assert_eq!(attrs.get("MATLAB_empty"), Some(&AttrValue::U64(1)));
}

#[test]
fn roundtrip_ascii_string_attribute() {
    let mut builder = FileBuilder::new();
    builder.set_attr("MATLAB_class", AttrValue::AsciiString("double".into()));
    builder.create_dataset("x").with_f64_data(&[0.0]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let attrs = file.root().attrs().unwrap();
    // ASCII strings are read as String
    assert_eq!(
        attrs.get("MATLAB_class"),
        Some(&AttrValue::String("double".into()))
    );
}

#[test]
fn roundtrip_complex32_convenience() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("z")
        .with_complex32_data(&[(1.0f32, 2.0f32), (3.0, 4.0)]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("z").unwrap();
    match ds.dtype().unwrap() {
        DType::Compound(fields) => {
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0], ("real".to_string(), DType::F32));
            assert_eq!(fields[1], ("imag".to_string(), DType::F32));
        }
        other => panic!("expected compound, got {other}"),
    }
    assert_eq!(ds.shape().unwrap(), vec![2]);
}

#[test]
fn roundtrip_complex64_convenience() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("z")
        .with_complex64_data(&[(1.0, -1.0), (0.0, 3.14)]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("z").unwrap();
    match ds.dtype().unwrap() {
        DType::Compound(fields) => {
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0], ("real".to_string(), DType::F64));
            assert_eq!(fields[1], ("imag".to_string(), DType::F64));
        }
        other => panic!("expected compound, got {other}"),
    }
}

#[test]
fn roundtrip_path_references() {
    // Simulate the MATLAB #refs# pattern:
    // Create a #refs# group with child datasets,
    // then create a reference dataset pointing to those children.
    let mut builder = FileBuilder::new();

    // Create #refs# group with two child datasets
    let mut refs_grp = builder.create_group("#refs#");
    refs_grp.create_dataset("child_a").with_f64_data(&[1.0, 2.0]);
    refs_grp.create_dataset("child_b").with_i32_data(&[10, 20, 30]);
    builder.add_group(refs_grp.finish());

    // Create a reference dataset pointing to those children by path
    builder
        .create_dataset("cell_refs")
        .with_path_references(&["#refs#/child_a", "#refs#/child_b"]);

    let bytes = builder.finish().unwrap();

    // Verify the file is valid
    let file = File::from_bytes(bytes).unwrap();

    // The referenced datasets should be readable
    let a = file.dataset("#refs#/child_a").unwrap();
    assert_eq!(a.read_f64().unwrap(), vec![1.0, 2.0]);

    let b = file.dataset("#refs#/child_b").unwrap();
    assert_eq!(b.read_i32().unwrap(), vec![10, 20, 30]);

    // The reference dataset should exist and have the right shape
    let refs = file.dataset("cell_refs").unwrap();
    assert_eq!(refs.shape().unwrap(), vec![2]);

    // Verify the reference dataset has the correct type
    assert_eq!(refs.dtype().unwrap(), DType::ObjectReference);
}

#[test]
fn roundtrip_path_references_2d_shape() {
    // MATLAB pattern: reference datasets with 2D shapes like [1, n] or [2, 1]
    let mut builder = FileBuilder::new();

    let mut refs_grp = builder.create_group("#refs#");
    refs_grp.create_dataset("a").with_f64_data(&[1.0]);
    refs_grp.create_dataset("b").with_f64_data(&[2.0]);
    refs_grp.create_dataset("c").with_f64_data(&[3.0]);
    refs_grp.create_dataset("d").with_f64_data(&[4.0]);
    builder.add_group(refs_grp.finish());

    // Shape [1, 4] — row vector of references
    builder
        .create_dataset("row_refs")
        .with_path_references(&["#refs#/a", "#refs#/b", "#refs#/c", "#refs#/d"])
        .with_shape(&[1, 4]);

    // Shape [2, 1] — column vector of references
    builder
        .create_dataset("col_refs")
        .with_path_references(&["#refs#/a", "#refs#/b"])
        .with_shape(&[2, 1]);

    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();

    let row = file.dataset("row_refs").unwrap();
    assert_eq!(row.shape().unwrap(), vec![1, 4]);
    assert_eq!(row.dtype().unwrap(), DType::ObjectReference);

    let col = file.dataset("col_refs").unwrap();
    assert_eq!(col.shape().unwrap(), vec![2, 1]);
    assert_eq!(col.dtype().unwrap(), DType::ObjectReference);
}

#[test]
fn roundtrip_varlen_ascii_array_attr() {
    let mut builder = FileBuilder::new();
    builder.set_attr(
        "MATLAB_fields",
        AttrValue::VarLenAsciiArray(vec!["x".into(), "y".into(), "z".into()]),
    );
    builder.create_dataset("x").with_f64_data(&[1.0]);
    let bytes = builder.finish().unwrap();

    // Verify we can read the file back
    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("x").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![1.0]);

    // Check that the MATLAB_fields attribute exists
    let attrs = file.root().attrs().unwrap();
    assert!(attrs.contains_key("MATLAB_fields"));
}

#[test]
fn valid_hdf5_signature() {
    let mut builder = FileBuilder::new();
    builder.create_dataset("x").with_f64_data(&[1.0]);
    let bytes = builder.finish().unwrap();

    assert!(bytes.len() > 8);
    assert_eq!(&bytes[..8], b"\x89HDF\r\n\x1a\n");
}

#[test]
fn roundtrip_empty_dataset_zero_dims() {
    let mut builder = FileBuilder::new();
    builder.set_attr("MATLAB_empty", AttrValue::U32(1));
    builder
        .create_dataset("empty")
        .with_dtype(make_f64_type())
        .with_shape(&[0, 0]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("empty").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![0, 0]);
    assert_eq!(ds.dtype().unwrap(), DType::F64);
    // Empty dataset should return empty data
    assert_eq!(ds.read_f64().unwrap(), vec![]);
}

#[test]
fn roundtrip_matlab_empty_shape_marker() {
    // MATLAB pattern: store original shape as u64 data with MATLAB_empty=1
    let matlab_dims: &[usize] = &[3, 4];
    let shape_data: Vec<u64> = matlab_dims.iter().map(|&d| d as u64).collect();

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_u64_data(&shape_data)
        .set_attr("MATLAB_class", AttrValue::AsciiString("double".into()))
        .set_attr("MATLAB_empty", AttrValue::U32(1));
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![2]); // 2-element 1D array
    assert_eq!(ds.dtype().unwrap(), DType::U64);
    assert_eq!(ds.read_u64().unwrap(), vec![3, 4]);
}

#[test]
fn roundtrip_matlab_refs_subsystem_pattern() {
    // Exercises the MATLAB v7.3 string/cell subsystem pattern:
    //   #refs# group   — holds the actual data as child datasets
    //   #subsystem#     — holds an MCOS dataset with path references into #refs#
    //   Some #refs# children themselves contain references to siblings

    let mut builder = FileBuilder::new();

    // --- #refs# group with several child datasets ---
    let mut refs_grp = builder.create_group("#refs#");

    // String-value datasets (simulating MATLAB string objects)
    refs_grp.create_dataset("a").with_u16_data(&[72, 101, 108, 108, 111]); // "Hello" UTF-16
    refs_grp.create_dataset("b").with_u16_data(&[87, 111, 114, 108, 100]); // "World"
    refs_grp.create_dataset("c").with_u16_data(&[70, 111, 111]); // "Foo"

    // A child dataset that itself references siblings (cross-references within #refs#)
    refs_grp
        .create_dataset("cell_data")
        .with_path_references(&["#refs#/a", "#refs#/b", "#refs#/c"])
        .with_shape(&[1, 3]);

    // Metadata dataset with attributes
    refs_grp
        .create_dataset("type_info")
        .with_u8_data(&[0, 0, 0, 0, 0, 0, 1, 0])
        .set_attr("MATLAB_class", AttrValue::AsciiString("string".into()));

    builder.add_group(refs_grp.finish());

    // --- #subsystem# group with MCOS dataset ---
    let mut subsys_grp = builder.create_group("#subsystem#");

    // MCOS dataset: array of references to objects in #refs#, shape [2, 1]
    subsys_grp
        .create_dataset("MCOS")
        .with_path_references(&["#refs#/cell_data", "#refs#/type_info"])
        .with_shape(&[2, 1]);

    builder.add_group(subsys_grp.finish());

    // --- Root-level dataset referencing into #refs# ---
    builder
        .create_dataset("data")
        .with_path_references(&["#refs#/a"])
        .with_shape(&[1, 1])
        .set_attr("MATLAB_class", AttrValue::AsciiString("string".into()));

    let bytes = builder.finish().unwrap();

    // --- Verify the file structure ---
    let file = File::from_bytes(bytes).unwrap();

    // Root children
    let root = file.root();
    let groups = root.groups().unwrap();
    assert!(groups.contains(&"#refs#".to_string()));
    assert!(groups.contains(&"#subsystem#".to_string()));

    // #refs# children are readable
    let a = file.dataset("#refs#/a").unwrap();
    assert_eq!(a.read_u16().unwrap(), vec![72, 101, 108, 108, 111]);
    let b = file.dataset("#refs#/b").unwrap();
    assert_eq!(b.read_u16().unwrap(), vec![87, 111, 114, 108, 100]);
    let c = file.dataset("#refs#/c").unwrap();
    assert_eq!(c.read_u16().unwrap(), vec![70, 111, 111]);

    // Cross-reference dataset in #refs# has correct 2D shape
    let cell = file.dataset("#refs#/cell_data").unwrap();
    assert_eq!(cell.shape().unwrap(), vec![1, 3]);
    assert_eq!(cell.dtype().unwrap(), DType::ObjectReference);

    // #subsystem#/MCOS has correct 2D shape
    let mcos = file.dataset("#subsystem#/MCOS").unwrap();
    assert_eq!(mcos.shape().unwrap(), vec![2, 1]);
    assert_eq!(mcos.dtype().unwrap(), DType::ObjectReference);

    // Root data dataset
    let data = file.dataset("data").unwrap();
    assert_eq!(data.shape().unwrap(), vec![1, 1]);
    assert_eq!(data.dtype().unwrap(), DType::ObjectReference);
}

#[test]
fn roundtrip_varlen_ascii_on_nested_group() {
    // MATLAB pattern: MATLAB_fields on a struct group (not root)
    let mut builder = FileBuilder::new();

    let mut grp = builder.create_group("my_struct");
    grp.create_dataset("x").with_f64_data(&[1.0, 2.0]);
    grp.create_dataset("y").with_f64_data(&[3.0, 4.0]);
    grp.set_attr("MATLAB_class", AttrValue::AsciiString("struct".into()));
    grp.set_attr(
        "MATLAB_fields",
        AttrValue::VarLenAsciiArray(vec!["x".into(), "y".into()]),
    );
    builder.add_group(grp.finish());

    let bytes = builder.finish().unwrap();
    let file = File::from_bytes(bytes).unwrap();

    // Verify group children
    let g = file.group("my_struct").unwrap();
    let datasets = g.datasets().unwrap();
    assert!(datasets.contains(&"x".to_string()));
    assert!(datasets.contains(&"y".to_string()));

    // Verify data
    assert_eq!(file.dataset("my_struct/x").unwrap().read_f64().unwrap(), vec![1.0, 2.0]);
    assert_eq!(file.dataset("my_struct/y").unwrap().read_f64().unwrap(), vec![3.0, 4.0]);
}

#[test]
fn roundtrip_group_only_no_datasets() {
    // MATLAB pattern: nested structs where a group has only sub-groups and attrs
    let mut builder = FileBuilder::new();

    let mut outer = builder.create_group("outer");
    outer.set_attr("MATLAB_class", AttrValue::AsciiString("struct".into()));

    let mut child_a = outer.create_group("a");
    child_a.create_dataset("val").with_f64_data(&[1.0]);
    child_a.set_attr("MATLAB_class", AttrValue::AsciiString("double".into()));
    outer.add_group(child_a.finish());

    let mut child_b = outer.create_group("b");
    child_b.create_dataset("val").with_i32_data(&[42]);
    child_b.set_attr("MATLAB_class", AttrValue::AsciiString("int32".into()));
    outer.add_group(child_b.finish());

    // outer has zero datasets, only sub-groups + attrs
    builder.add_group(outer.finish());

    let bytes = builder.finish().unwrap();
    let file = File::from_bytes(bytes).unwrap();

    let outer_grp = file.group("outer").unwrap();
    let groups = outer_grp.groups().unwrap();
    assert!(groups.contains(&"a".to_string()));
    assert!(groups.contains(&"b".to_string()));
    assert_eq!(outer_grp.datasets().unwrap().len(), 0);

    assert_eq!(file.dataset("outer/a/val").unwrap().read_f64().unwrap(), vec![1.0]);
    assert_eq!(file.dataset("outer/b/val").unwrap().read_i32().unwrap(), vec![42]);
}
