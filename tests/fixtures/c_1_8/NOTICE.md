# HDF5 1.8-era read fixtures

`v1_superblock.h5` and `v2_superblock.h5` are written by **HDF5 1.8.23**, the
last 1.8 release, and committed as ground-truth read fixtures for
`tests/c_1_8_read_compat.rs` and the `parse_v1_against_a_c_written_superblock`
unit test in `src/superblock.rs`.

They are produced by `regen.c` in this directory, built and run by `regen.sh`
against the 1.8.23 install that `scripts/check-hdf5-18.sh` leaves under
`tmp/hdf5-18-check/`. Nothing in `cargo test` invokes either: the fixtures are
committed, so the tests that read them are ordinary tests with no external
dependency and run wherever `cargo test` does, 32-bit targets included.

## Why an old library, and why committed

The suite already reaches old formats two ways, and both miss what these cover.

The `hdf5-metno` dev-dependency builds a *current* libhdf5, which can be asked
for an older format with `H5Pset_libver_bounds`. That is not the same artifact
as an old library writing it — the modern library's encoder is what produces the
bytes either way — and every test using it is gated off 32-bit targets, because
`hdf5-metno` requires 64-bit pointers. Address arithmetic is exactly what breaks
on 32-bit, so the gate removes the coverage where it is most wanted.

The committed fixtures in `tests/fixtures/` cover superblock 0 (13 files) and
superblock 3 (10 files), and nothing between. These two fill that gap:

| file | superblock | why it is hard to get elsewhere |
| --- | --- | --- |
| `v1_superblock.h5` | 1 | The C library writes version 1 **only** when a B-tree K value is non-default. Nothing else in the corpus is a version 1 superblock at all. |
| `v2_superblock.h5` | 2 | 1.8's newest format, and what this crate now writes by default for `.mat` files. A `.mat` a C-based tool has touched comes back looking like this. |

## What `v1_superblock.h5` settles

The version 1 superblock adds three B-tree K values, and a defect fixed in
0.33.0 read two of them from the wrong offsets — the chunk B-tree K where the
status flags belong, so a file with the C library's default K of 32 read back as
"held by a writer" and every open refused it.

`src/superblock.rs` tested `parse_v1` against `build_v1_bytes`, which the same
hand wrote from the same reading of the specification. Those two agree about the
field order whether or not that order is right, which is what let the defect
exist. Reproducing it — laying the bytes out the wrong way in *both* the builder
and the parser — leaves both hand-built tests passing and fails only the test
that reads this file.

The file is written with `H5Pset_sym_k(8, 16)` and `H5Pset_istore_k(64)`, so all
three K values are distinct from one another and from the library's defaults,
and any permutation of the three reads back wrong.

## Contents

Both files hold the same objects, so a difference between them is the format:

- `/values` — contiguous `f64[4]`, with a `units` string attribute
- `/chunked` — `i32[1000]` in 100-element chunks, deflate level 6. Under a
  pre-1.10 superblock this is indexed by a version 1 B-tree, which this crate
  reads and does not write, so nothing it produces can stand in for it.
- `/grp` — a group with a `tag` attribute, holding `/grp/inner`
- `/` — a `root_attr` string attribute

## Licensing

Nothing here is vendored from the HDF5 distribution. `regen.c` is this project's
own code, and the `.h5` files are its output. HDF5 1.8.23 is used as a build
tool, the way a compiler is; it is not redistributed, and its own test files are
deliberately not copied — they exercise libhdf5 features this crate does not
support, so most would be refusals rather than coverage.
