# Variable-Length Strings

HDF5 strings come in two on-disk shapes: fixed-length (a fixed byte width per element, padded) and variable-length (each element stores a pointer into the file's global heap). This page covers reading both, with a focus on bounding and streaming variable-length reads so a single dataset cannot exhaust memory before you know how big it is.

For the broader reading API see [Reading](reading.md). For the streaming backend used in several examples here, see [Streaming large files](streaming.md). Writing variable-length strings as *attributes* is a separate facility (the `AttrValue::VarLenString` family), covered in [Groups & attributes](groups-attributes.md).

## Reading any string dataset

`Dataset::read_string` reads both fixed-length and variable-length string datasets and returns a `Vec<String>`. You do not need to know which encoding the file uses: the method inspects the datatype and dispatches accordingly.

```rust
use hdf5_pure::File;

let file = File::open("strings.h5").unwrap();
let names = file.dataset("names").unwrap().read_string().unwrap();

for name in &names {
    println!("{name}");
}
```

This is the right call for ordinary datasets whose size you trust. For large variable-length datasets, the methods below let you measure and bound the read first.

## Measuring payload size before allocation

A variable-length string dataset's in-memory size is not implied by its shape: each element points at a separately sized run of bytes in the global heap. `Dataset::vlen_string_payload_size` reports the total payload size, in bytes, that a full read would materialize, without building the `Vec<String>`:

```rust
use hdf5_pure::File;

let file = File::open("strings.h5").unwrap();
let dataset = file.dataset("names").unwrap();

let payload_bytes = dataset.vlen_string_payload_size().unwrap();
println!("{payload_bytes} bytes of string payload");
```

!!! note
    The payload figure counts only the bytes referenced by the variable-length elements. It excludes the `Vec<String>` and `String` allocation metadata that Rust adds when the strings are actually materialized. This mirrors the role of HDF5's `H5Dvlen_get_buf_size`.

`vlen_string_payload_size` is defined only for variable-length string datasets; called on any other datatype it returns an error rather than a meaningless number.

## Bounding a read with `VlenStringReadOptions`

When a dataset may be larger than you are willing to allocate, pass `VlenStringReadOptions` to bound the read. Both limits are checked *before* any string payload is materialized, so an oversized dataset fails fast instead of allocating first.

| Builder method | Bounds | Notes |
|---|---|---|
| `VlenStringReadOptions::new()` | — | No limits (equivalent to the default). |
| `.with_max_elements(n)` | Number of VL elements | Checked against the dataspace element count. |
| `.with_max_payload_bytes(n)` | Total referenced payload bytes | Excludes Rust container metadata. |

The options are a `const`-constructible builder, so a limit can be assembled in a `const` context:

```rust
use hdf5_pure::{File, VlenStringReadOptions};

let file = File::open("strings.h5").unwrap();
let dataset = file.dataset("names").unwrap();

let options = VlenStringReadOptions::new()
    .with_max_elements(100_000)
    .with_max_payload_bytes(64 * 1024 * 1024);

let names = dataset.read_vlen_strings(options).unwrap();
```

`Dataset::read_vlen_strings` reads the whole dataset into a `Vec<String>` like `read_string`, but enforces the limits. If either bound is exceeded the call returns an error and nothing is allocated for the payload.

## Visiting strings one at a time

To process a large dataset without ever holding every decoded string at once, use `Dataset::visit_vlen_strings`. It takes the same `VlenStringReadOptions` and a closure that is invoked once per element with a `&str` borrowed for the duration of the call:

```rust
use hdf5_pure::{File, VlenStringReadOptions};

let file = File::open_streaming("strings.h5").unwrap();
let dataset = file.dataset("names").unwrap();

let mut longest = 0usize;
dataset.visit_vlen_strings(VlenStringReadOptions::new(), |value| {
    longest = longest.max(value.len());
}).unwrap();

println!("longest string: {longest} bytes");
```

!!! tip
    Because the slice handed to the closure is valid only for that call, this is the lowest-memory way to scan a variable-length dataset: aggregate, filter, or stream each string out as it arrives rather than collecting them. `read_vlen_strings` is implemented on top of `visit_vlen_strings` by pushing each value onto a `Vec`.

## How payloads are fetched

Variable-length elements reference objects in the file's global heap, and many elements typically share a heap collection. The reader indexes each shared global heap collection once per read and then fetches only the object payloads the dataset's elements actually reference, so duplicated or shared storage is not read repeatedly.

This holds for both backends. Everything on this page works the same with `File::open` (whole file in memory) and `File::open_streaming` (metadata and chunks fetched on demand), so the bounding and streaming-visit pattern is exactly what you want for a multi-gigabyte file opened with the streaming reader.

## Scope

These methods cover *reading* strings. On the write side, `DatasetBuilder::with_ascii_strings` and `with_strings` write the fixed-width kind (see [Writing](writing.md#strings)); the variable-length kind has two facilities: `DatasetBuilder::with_vlen_strings` writes a contiguous variable-length UTF-8 string dataset, and the attribute variants `AttrValue::VarLenString`, `VarLenStringArray`, `VarLenAsciiString` and `VarLenAsciiStringArray` write a variable-length string as an attribute, with `AttrValue::VarLenAsciiArray` writing MATLAB's sequence-of-one-byte-strings shape instead (see [Groups & attributes](groups-attributes.md)). Both of those store their payloads in the file's global heap, split across as many collections as their element count needs. Chunked, filtered, and resizable variable-length-string datasets write too: their heap collections are placed ahead of the chunk data so the element references carry real addresses before a chunk is compressed. An existing dataset's strings can also be replaced in place through `Dataset::write_staged` with the same builder method, which places a fresh collection for the new strings; see [Editing](editing.md). For how string datatypes map between HDF5 and this crate, see [Data types](../reference/data-types.md).
