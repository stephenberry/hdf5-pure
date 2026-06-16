# Groups & Attributes

HDF5 files are hierarchical: datasets live inside groups, groups nest inside other groups, and any object (the root, a group, or a dataset) can carry typed metadata in the form of attributes. This page covers building a nested hierarchy with the write API, attaching attributes of several types, and walking the structure back when reading.

!!! tip "Runnable example"
    A complete, runnable version of everything on this page lives in
    [`examples/groups_and_attributes.rs`](https://github.com/stephenberry/hdf5-pure/blob/main/examples/groups_and_attributes.rs).
    Run it with:

    ```bash
    cargo run --example groups_and_attributes
    ```

## Building a hierarchy

Groups are created with a builder API. `FileBuilder::create_group(name)` returns a `GroupBuilder` for a top-level group, and `GroupBuilder::create_group(name)` returns a nested `GroupBuilder` for a sub-group. Each group builder can hold datasets (via `create_dataset`), attributes (via `set_attr`), and further sub-groups.

A group builder is not part of the file until it is finished and attached: call `finish()` to turn it into a `FinishedGroup`, then pass that to its parent's `add_group()`. For a sub-group, the parent is another `GroupBuilder`; for a top-level group, the parent is the `FileBuilder`.

```rust
use hdf5_pure::{AttrValue, File, FileBuilder};

let mut builder = FileBuilder::new();

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

// Attach the top-level group to the file.
builder.add_group(sensors.finish());
```

The pattern is uniform at every level: build, finish, attach. A child must be finished and added before its parent is finished, since `finish()` consumes the builder.

See the [writing guide](writing.md) for the full dataset builder API used inside groups.

## Attributes

Attributes are small named pieces of metadata. Set them on the root via `FileBuilder::set_attr`, on a group via `GroupBuilder::set_attr`, or on a dataset via `DatasetBuilder::set_attr` (the dataset form is chainable and returns `&mut Self`). The value is an `AttrValue`, an enum covering the common scalar, array, and string encodings:

```rust
// Root-level metadata.
builder.set_attr("title", AttrValue::String("experiment 7".into()));
builder.set_attr("run", AttrValue::I64(7));
builder.set_attr("calibration", AttrValue::F64Array(vec![0.1, 0.2, 0.3]));
```

The `AttrValue` variants and their HDF5 encodings are:

| Variant | HDF5 encoding |
|---|---|
| `AttrValue::F64` | 64-bit float scalar |
| `AttrValue::F64Array` | 64-bit float array |
| `AttrValue::I32` | Signed 32-bit integer scalar |
| `AttrValue::I64` | Signed 64-bit integer scalar |
| `AttrValue::I64Array` | Signed 64-bit integer array |
| `AttrValue::U32` | Unsigned 32-bit integer scalar |
| `AttrValue::U64` | Unsigned 64-bit integer scalar |
| `AttrValue::String` | UTF-8 string (null-padded) |
| `AttrValue::StringArray` | Array of UTF-8 strings |
| `AttrValue::AsciiString` | Fixed-width ASCII string (charset = ASCII) |
| `AttrValue::AsciiStringArray` | Array of fixed-width ASCII strings (null-padded to the longest element) |
| `AttrValue::VarLenAsciiArray` | Array of variable-length ASCII strings (uses a global heap collection) |

!!! note
    `AttrValue::AsciiString`, `AttrValue::AsciiStringArray`, and `AttrValue::VarLenAsciiArray` exist for compatibility with MATLAB and matio, which expect fixed-width or variable-length ASCII rather than UTF-8 for certain conventional attributes. See the [data types reference](../reference/data-types.md) for the full type mapping.

## Reading the hierarchy back

Open the file and start from `File::root()`, which returns a `Group` for the root. From any `Group` you can list its contents and read its attributes:

- `groups()` returns the names of child groups (`Vec<String>`).
- `datasets()` returns the names of child datasets (`Vec<String>`).
- `attrs()` returns the attributes as a `HashMap<String, AttrValue>`.

```rust
let file = File::from_bytes(builder.finish().unwrap()).unwrap();

let root = file.root();
let root_attrs = root.attrs().unwrap(); // HashMap<String, AttrValue>

let sensors = file.group("sensors").unwrap();
println!("child groups: {:?}", sensors.groups().unwrap());   // ["imu"]
println!("datasets:     {:?}", sensors.datasets().unwrap()); // ["humidity", "pressure"]
println!("attributes:   {:?}", sensors.attrs().unwrap());
```

`File::group(path)` resolves a group by path, and `Group::group(name)` resolves a child relative to that group. The names returned by `groups()` and `datasets()` are not sorted in any guaranteed order, so sort them yourself if you need a stable listing.

### Addressing datasets

Datasets are addressable two ways: by full path from the file, or by name from their parent group. Both resolve to the same dataset.

```rust
// Full path from the file.
let accel = file.dataset("sensors/imu/accel").unwrap();
println!("{:?}", accel.read_f64().unwrap()); // [0.0, 0.0, 9.81]

// By name, relative to the parent group.
let imu = file.group("sensors/imu").unwrap();
let accel = imu.dataset("accel").unwrap();
```

See the [reading guide](reading.md) for the dataset read API (`read_f64`, `shape`, and friends).

## MATLAB struct convention

The ASCII attribute variants exist primarily so that groups can follow MATLAB's struct convention: a struct is a group carrying `MATLAB_class = "struct"` and a `MATLAB_fields` attribute (typically `AttrValue::VarLenAsciiArray`) listing the field names, with each field stored as a child dataset tagged with its own `MATLAB_class`. See the [MATLAB interop page](../interop/matlab.md) for the full convention and worked examples.
