//! Building a nested group hierarchy with attributes of several types, then
//! walking it back: listing child groups and datasets and reading attributes.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example groups_and_attributes
//! ```

use hdf5_pure::{AttrValue, File, FileBuilder};

fn main() {
    let mut builder = FileBuilder::new();

    // Root-level metadata. Attributes come in many flavors; a representative
    // spread is shown here.
    builder.set_attr("title", AttrValue::String("experiment 7".into()));
    builder.set_attr("run", AttrValue::I64(7));
    builder.set_attr("calibration", AttrValue::F64Array(vec![0.1, 0.2, 0.3]));

    // A group with its own datasets and attributes.
    let mut sensors = builder.create_group("sensors");
    sensors.set_attr("location", AttrValue::AsciiString("lab_a".into()));
    sensors.set_attr("channels", AttrValue::I64Array(vec![0, 1, 2]));
    sensors
        .create_dataset("pressure")
        .with_f32_data(&[101.3, 101.5, 101.4]);
    sensors
        .create_dataset("humidity")
        .with_f32_data(&[40.0, 41.5]);

    // A nested sub-group. Build it, finish it, and attach it to its parent.
    let mut imu = sensors.create_group("imu");
    imu.set_attr("model", AttrValue::String("MPU-9250".into()));
    imu.create_dataset("accel").with_f64_data(&[0.0, 0.0, 9.81]);
    sensors.add_group(imu.finish());

    builder.add_group(sensors.finish());

    let file = File::from_bytes(builder.finish().unwrap()).unwrap();

    // ---- Walk the hierarchy --------------------------------------------
    let root = file.root();
    println!("root attributes:");
    print_attrs(&root.attrs().unwrap());

    let sensors = file.group("sensors").unwrap();
    println!("\n/sensors");
    println!("  child groups:   {:?}", sorted(sensors.groups().unwrap()));
    println!(
        "  datasets:       {:?}",
        sorted(sensors.datasets().unwrap())
    );
    println!("  attributes:");
    print_attrs_indented(&sensors.attrs().unwrap(), "    ");

    // Datasets are addressable by full path from the file, or by name from
    // their parent group.
    let accel = file.dataset("sensors/imu/accel").unwrap();
    println!("\n/sensors/imu/accel = {:?}", accel.read_f64().unwrap());

    assert_eq!(
        sorted(sensors.datasets().unwrap()),
        ["humidity", "pressure"]
    );
    assert_eq!(sensors.groups().unwrap(), ["imu"]);
    assert_eq!(accel.read_f64().unwrap(), vec![0.0, 0.0, 9.81]);
    println!("\nhierarchy verified");
}

fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v
}

fn print_attrs(attrs: &std::collections::HashMap<String, AttrValue>) {
    print_attrs_indented(attrs, "  ");
}

fn print_attrs_indented(attrs: &std::collections::HashMap<String, AttrValue>, indent: &str) {
    let mut names: Vec<_> = attrs.keys().collect();
    names.sort();
    for name in names {
        println!("{indent}{name} = {:?}", attrs[name]);
    }
}
