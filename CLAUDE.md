# Project guidance

## Build hygiene

Run `cargo clean -p hdf5-pure` when builds start taking tens of seconds before any work appears to happen. Incremental compilation leaves one object file per codegen unit in `target/*/deps` on every rebuild, and cargo never reaps them. Cargo and rustc scan `deps/`, so once that directory holds hundreds of thousands of files the listing alone costs tens of seconds on every build. Measured here at 723k leftover objects: an incremental rebuild of the library plus one integration test took 18.3s, against 1.3s in a clean target directory. Measured again at 287k: a full `cargo test --all-features` was still on its 10th of 98 test binaries after twelve minutes; the clean removed 448k files and 49 GiB in 82 seconds.

`[profile.dev]` and `[profile.test]` set `codegen-units = 4` against rustc's default of 256, which cuts objects per full rebuild from ~9,800 to ~600 for about one second on a library build. That slows the accumulation; it does not stop it, so the periodic clean is still the remedy. The reasoning and the measured curve are in `Cargo.toml` beside the setting.

`cargo clean -p hdf5-pure` drops this package's artifacts across every profile and feature set, including all 97 integration binaries, and leaves dependencies (and the compiled libhdf5) alone. It costs one non-incremental rebuild of the crate, about five seconds. `CARGO_INCREMENTAL=0` stops the accumulation outright, but it makes every rebuild of the library non-incremental — 4.6s against 0.9s — so the periodic clean is the better trade.

Prefer `cargo test --lib` for the fast loop: the ~580 library tests run in about two seconds, and the mutation checks that prove a new test is load-bearing belong there. `cargo test --all-features` builds 97 separate integration binaries and compiles libhdf5 from source through `hdf5-metno-src`, so treat it as a pre-push gate rather than an inner-loop command, and keep to one feature set locally — every distinct set is a full parallel copy of the build graph.

## The HDF5 1.8 output format

`scripts/check-hdf5-18.sh` builds HDF5 1.8.23 and points its tools at both formats this crate writes. **Run it when changing anything about superblock or object-header message versions**, and read it before assuming the test suite covers that ground — it does not, and cannot.

Which HDF5 library MATLAB links has changed across releases, and MathWorks documents it: **1.8.12 before R2021b**, 1.10.7 in R2021b, 1.10.x through R2024a, 1.14.4.3 since R2024b. A version 3 superblock is a 1.10 addition, so a `.mat` carrying one cannot be opened at all before R2021b. That is why `mat::Options::libver` defaults to `LibVer::V18` and why `FileBuilder::with_libver_bounds` selects a format rather than merely validating one. Around R2021b MathWorks also shipped 1.8.12 on the MAT path while `h5read` used 1.10.7, which is the split behind the reported symptom of `h5disp` working where `load` fails; its duration is undocumented, so the 1.8 default is the safe choice on newer releases too.

No test can catch a regression here. The `hdf5-metno` dev-dependency builds a current libhdf5, `h5py` links a current libhdf5, and every third-party MAT v7.3 reader (`mat73`, `hdf5storage`, `pymatreader`, MAT.jl, matio) delegates to whichever libhdf5 *it* links. All of them read a version 3 superblock without complaint. The failure is one byte below everything they parse, so only an old library can see it, and none is available as a dev-dependency. The script needs network access and a couple of minutes on its first run, which is why it is a command you run rather than a CI job.

What it measured when the 1.8 format landed in 0.34.0, against 1.8.23:

| file | HDF5 1.8.23 `h5dump` |
| --- | --- |
| superblock 3 (the default through 0.33.0) | `unable to open file`, exit 1 |
| superblock 2 (the 1.8 format) | reads data, groups, and every attribute |

1.8 does not degrade or warn on the newer superblock — it cannot open the file at all. The script also round-trips both fixtures through 1.8's `h5repack` and counts attributes, which is the other half of the same story: before the Attribute Info fix, `h5repack` copied every object with none of its attributes.

One limit worth stating: this proves the 1.8 *format* boundary, not that a particular MathWorks build accepts the file. `examples/octave/check_format.m` asks MATLAB directly, against a pair of files differing only in that format, and reports which of the outcomes it got; run it under real MATLAB, since Octave's HDF5 is modern and reads both. `verify.m` covers content rather than format and passes under Octave.

## Changelog

Keep `CHANGELOG.md` entries concise and reader-facing. Each entry is one or two sentences: lead with the user-facing capability and the public API name, keep at most one short caveat clause naming what is still refused or limited, and end with the issue/PR link in `([#NN](url))` form. Use a **Breaking:** prefix for breaking changes.

Drop the things that bloat it: bit/byte-level internals and root-cause mechanics, "validated/verified against the reference C library" narration, "byte-for-byte" boilerplate, exhaustive enumerations of every refused case, and development cross-references like "addressed below". Commit `d5a966b` (#61, "Trim the changelog to concise, reader-facing entries") is the canonical example of the before/after; match its `before -> after` for any new entry, and don't let the `[Unreleased]` section regrow into PR-description-length paragraphs.

`docs/reference/changelog.md` is a generated include of `CHANGELOG.md`, so edit only the root file.

## Versioning

The version in `Cargo.toml` is the version being *developed*, not the one last published. **A PR that breaks the public API bumps it to the next minor (`0.24.0` -> `0.25.0`), and `Cargo.lock` with it, in that same PR**, alongside its `CHANGELOG.md` entry. If it already reads the next minor, leave it: one bump covers every breaking change in the cycle. Additive-only PRs need no bump, and the release takes care of theirs.

This is what keeps the `SemVer (cargo-semver-checks)` job green. That job compares the branch against the latest crates.io release and fails a breaking change unless the manifest already declares a bump large enough to carry it. Deferring the bump to release time makes every breaking PR red by construction, and a check that is expected to be red stops being read at all, including when it flags an *unintended* break.

Know the cost: once the manifest declares the bump, `cargo-semver-checks` reports "no semver update required" and **skips every check** for the rest of the cycle, so it catches an unintended break only in the first breaking PR of a cycle (and throughout an additive-only one). `scripts/release.sh` compensates by running the full check once at release time with `--release-type minor`, which overrides that derivation; read its output against the `[Unreleased]` section you are about to promote. An empty report on a cycle whose changelog claims a breaking change is the signal worth chasing.

`scripts/release.sh` reads the previous version from the latest `vX.Y.Z` tag, so it accepts an already-bumped manifest. It refuses a version that does not come after the last release, a manifest outside `last-release .. version-being-cut`, and a changelog a previous run already promoted.
