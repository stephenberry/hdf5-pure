# Project guidance

## Build hygiene

Run `cargo clean -p hdf5-pure` when builds start taking tens of seconds before any work appears to happen. Incremental compilation leaves one object file per codegen unit in `target/*/deps` on every rebuild, and cargo never reaps them. Cargo and rustc scan `deps/`, so once that directory holds hundreds of thousands of files the listing alone costs tens of seconds on every build. Measured here at 723k leftover objects: an incremental rebuild of the library plus one integration test took 18.3s, against 1.3s in a clean target directory. Measured again at 287k: a full `cargo test --all-features` was still on its 10th of 98 test binaries after twelve minutes; the clean removed 448k files and 49 GiB in 82 seconds.

`[profile.dev]` and `[profile.test]` set `codegen-units = 4` against rustc's default of 256, which cuts objects per full rebuild from ~9,800 to ~600 for about one second on a library build. That slows the accumulation; it does not stop it, so the periodic clean is still the remedy. The reasoning and the measured curve are in `Cargo.toml` beside the setting.

`cargo clean -p hdf5-pure` drops this package's artifacts across every profile and feature set, including all ~110 integration binaries, and leaves dependencies (and the compiled libhdf5) alone. It costs one non-incremental rebuild of the crate, about five seconds. `CARGO_INCREMENTAL=0` stops the accumulation outright, but it makes every rebuild of the library non-incremental — 4.6s against 0.9s — so the periodic clean is the better trade.

Prefer `cargo test --lib` for the fast loop: the ~580 library tests run in about two seconds, and the mutation checks that prove a new test is load-bearing belong there. The full suite builds ~110 separate integration binaries and compiles libhdf5 from source through `hdf5-metno-src`, so treat it as a pre-push gate rather than an inner-loop command, and keep to one feature set locally — every distinct set is a full parallel copy of the build graph.

## The pre-push gate

```
cargo nextest run --all-features && cargo test --all-features --doc
```

Both halves are needed. **nextest does not run doctests** and never will: `rustdoc` compiles those, not the test harness. Running only the first command leaves 46 doctests silently unexecuted.

**Neither half runs rustdoc's link lints.** Doctests compile the code inside ```` ```rust ```` blocks; they say nothing about the links around them. Add a third command whenever a public doc comment changed:

```
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --features "provenance zfp ndarray serde"
```

It takes about two seconds and it is the exact command the `Lint (fmt, clippy)` job runs. The lint that bites is `rustdoc::private_intra_doc_links`, and the curated API makes this crate unusually easy to trip: a module is `pub(crate)` while individual items inside it are `pub` and re-exported from `lib.rs`, so a link from one re-exported item to a neighbour that was *not* re-exported resolves fine for you and not at all for a reader of the published docs. `ChunkCacheConfig::from_h5p_cache` linking `[`CachePass`]` reddened that job on PR #375 while the other 22 passed. The same link written in the docs of a *private* item is accepted, since only public documentation is checked — which is what makes the bad one easy to write by analogy with its neighbours.

nextest rather than `cargo test` because `cargo test` runs the ~110 integration binaries one after another, so the suite spends most of its wall clock on a single core no matter how many the machine has. Measured on a 10-core host at `--all-features`: **46.1s under `cargo test` (115% CPU) against 17.9s under nextest (421% CPU)**, running the identical 2,034 tests. Install it with `cargo install cargo-nextest --locked`; CI runs the same two commands per matrix config.

`.config/nextest.toml` fails the run on any test slower than 90s and prints a warning at 30s. That gate exists because a single test once spent 117s issuing one `fsync` per appended element — five times the rest of the suite combined — and nothing surfaced it, since `cargo test` reports no per-test timings at all. **A long-running test in this suite is a defect signal, not a cost of doing business**: the usual cause is a session left on the default `SyncPolicy::Always` while looping over appends, and the fix is `SyncPolicy::OnClose`, which writes byte-identical files (`tests/sync_policy.rs` asserts exactly that) and still barriers at `close`. Reach for a per-test override in that config only after ruling this out.

## Allocation gates

Two test binaries install [`heapscope`](https://crates.io/crates/heapscope) as their `#[global_allocator]` and measure what the read and write paths allocate (issue #228). They answer different questions and fail for different reasons.

`tests/allocation_bounds.rs` states **rules**: a windowed read allocates on the order of its window, a chunked read and a chunked write each cost a small constant number of allocations per chunk, a windowed variable-length read does not allocate per object of the heap collection it walks and crosses that collection about once rather than repeatedly, and a filtered read or write builds its zlib codec once rather than per chunk (that last one was worth 743 MB of allocation on an 8 MiB write). Every one is a claim about how the work scales, which is what lets it hold across platforms where an exact count would not. It runs in every `test`-matrix job on all three operating systems (not in the `cross` i686/s390x jobs, which name their test targets explicitly).

Each bound is paired with a **floor** — usually "the read's own output is in this measurement". Regions are per thread, so work that moves onto a worker leaves every figure near zero and every ceiling passing; the floor is what makes that fail instead. The one bound deliberately close to its measurement is `BLOCKS_PER_CHUNK`, about 3% above a rate that converges to 3.00 per chunk: one extra allocation in the per-chunk loop is the defect it exists to catch, so it is tight on purpose.

`tests/allocation_baseline.rs` pins the **numbers**, in `tests/baselines/`, so a 5% creep that passes every rule still arrives in a pull request as a line a reviewer reads. An exact count is a property of one target, one toolchain *and* one feature set — `--all-features` measured 416 bytes above the default set on the same commit — so it compiles only under the crate's default features plus `heap-baseline`, and runs in one CI job on aarch64 macOS, the host the numbers were measured on. The `--all-features` pre-push gate therefore does not run it; this does:

```
cargo nextest run --features heap-baseline --test allocation_baseline
```

Re-record with `HEAPSCOPE_UPDATE_BASELINE=1 cargo test --features heap-baseline --test allocation_baseline` and commit the diff with the change that moved it, and pin the job's toolchain deliberately — a `std` release that moves a figure would otherwise redden every open PR at once. The comparison is one-sided (only figures that *grew* are reported), so an improvement never fails and never appears: re-record after one too, or the old worse number stays as headroom a later regression spends for free.

`examples/heap_profile.rs` is the third piece and the one to reach for first when a bound fails: `cargo run --release --example heap_profile` writes `target/heap-profile.html` and prints the heaviest call sites, broken down by phase (write chunked, read whole, read window, ...). Run it in release — a debug build's counts are the same but its stacks are full of the inlining that did not happen.

**On x86_64 this needs frame pointers**, which `.cargo/config.toml` sets for `x86_64-unknown-linux-gnu` and `x86_64-apple-darwin`. aarch64 and Windows need no flag. A shell that exports `RUSTFLAGS` replaces those flags rather than adding to them, and the two binaries then refuse to start with a message naming the file — deliberately, because a heap gate that cannot measure must not pass.

Measurement is per region and per thread, so these tests share a process without a lock between them: what `allocation::measure` reports is what the calling thread allocated inside the call, and nothing else. One limit is worth knowing before adding a *baseline*: `heapscope::HeapStats` is `#[non_exhaustive]`, so a baseline can only be taken of a whole process, which is why that file holds exactly one `#[test]` and why its fixture is part of the measurement.

## The HDF5 1.8 output format

`scripts/check-hdf5-18.sh` builds HDF5 1.8.23 and points its tools at both formats this crate writes. **Run it when changing anything about superblock or object-header message versions**, and read it before assuming the test suite covers that ground — it does not, and cannot.

Which HDF5 library MATLAB links has changed across releases, and MathWorks documents it: **1.8.12 in R2021a and earlier**, 1.10.7 in R2021b, 1.10.8 through 1.10.11 across R2022a–R2024a, 1.14.4.3 since R2024b. A version 3 superblock is a 1.10 addition, so a `.mat` carrying one cannot be opened at all before R2021b. That is why `mat::Options::libver` defaults to `LibVer::V18` and why `FileBuilder::with_libver_bounds` selects a format rather than merely validating one.

**The linked library version does not predict what `load` accepts, and the gap is not small.** Measured on **R2023a Update 1**, whose `H5.get_libversion` reports **1.10.8**: the version 2 superblock loads and decodes correctly, and the version 3 one fails outright with "Not a binary MAT-file" — from a library that reads a version 3 superblock without difficulty. The two files were byte-identical through the 512-byte userblock, differing in 35 bytes that are all version and checksum, so the format is the only variable. MathWorks documents the start of this (around R2021b it shipped 1.8.12 on the MAT path while `h5read` used 1.10.7, the split behind the symptom of `h5disp` working where `load` fails) but not its extent; the measurement puts it at R2023a, two years later. Whether the cause is the older library or a MAT reader that caps the superblock version is undetermined and does not change the conclusion: the 1.8 default is required on releases the version table would call safe, not merely prudent.

`examples/octave/check_format.m` produced that measurement and prints its own inputs, so a run on another release extends the record. Its `CONFIRMED` verdict calls out the 1.10-reported-but-refused case specifically, since that is the one the published table does not predict.

No test can catch a regression here. The `hdf5-metno` dev-dependency builds a current libhdf5, `h5py` links a current libhdf5, and every third-party MAT v7.3 reader (`mat73`, `hdf5storage`, `pymatreader`, MAT.jl, matio) delegates to whichever libhdf5 *it* links. All of them read a version 3 superblock without complaint. The failure is one byte below everything they parse, so only an old library can see it, and none is available as a dev-dependency. The script needs network access and a couple of minutes on its first run, which is why it is a command you run rather than a CI job.

What it measured when the 1.8 format landed in 0.34.0, against 1.8.23:

| file | HDF5 1.8.23 `h5dump` |
| --- | --- |
| superblock 3 (the default through 0.33.0) | `unable to open file`, exit 1 |
| superblock 2 (the 1.8 format) | reads data, groups, and every attribute |

1.8 does not degrade or warn on the newer superblock — it cannot open the file at all. The script also round-trips both fixtures through 1.8's `h5repack` and counts attributes, which is the other half of the same story: before the Attribute Info fix, `h5repack` copied every object with none of its attributes.

The "reads data" row is a claim the script now checks rather than one a reader has to take on trust: it compares dataset and attribute *values* against what `examples/libver_fixtures.rs` wrote, not just `h5dump`'s exit status. `h5dump -n` lists object names, so a file whose headers all decode while its data resolves to the wrong offset — the base-address defect class — opens cleanly and lists everything, and an exit-status check would call it passing.

One limit worth stating: this proves the 1.8 *format* boundary, not that a particular MathWorks build accepts the file. That second question is what `check_format.m` answers, and R2023a answered it above — but one release is one data point, so run it on whatever MATLAB is to hand. Octave cannot substitute: its HDF5 is modern and reads both formats, so it reports INCONCLUSIVE by design rather than guessing. `verify.m` covers content rather than format and does pass under Octave.

## Changelog

Keep `CHANGELOG.md` entries concise and reader-facing. Each entry is one or two sentences: lead with the user-facing capability and the public API name, keep at most one short caveat clause naming what is still refused or limited, and end with the issue/PR link in `([#NN](url))` form. Use a **Breaking:** prefix for breaking changes.

Drop the things that bloat it: bit/byte-level internals and root-cause mechanics, "validated/verified against the reference C library" narration, "byte-for-byte" boilerplate, exhaustive enumerations of every refused case, and development cross-references like "addressed below". Commit `d5a966b` (#61, "Trim the changelog to concise, reader-facing entries") is the canonical example of the before/after; match its `before -> after` for any new entry, and don't let the `[Unreleased]` section regrow into PR-description-length paragraphs.

`docs/reference/changelog.md` is a generated include of `CHANGELOG.md`, so edit only the root file.

## Versioning

The version in `Cargo.toml` is the version being *developed*, not the one last published. **A PR that breaks the public API bumps it to the next minor (`0.24.0` -> `0.25.0`), and `Cargo.lock` with it, in that same PR**, alongside its `CHANGELOG.md` entry. If it already reads the next minor, leave it: one bump covers every breaking change in the cycle. Additive-only PRs need no bump, and the release takes care of theirs.

This is what keeps the `SemVer (cargo-semver-checks)` job green. That job compares the branch against the latest crates.io release and fails a breaking change unless the manifest already declares a bump large enough to carry it. Deferring the bump to release time makes every breaking PR red by construction, and a check that is expected to be red stops being read at all, including when it flags an *unintended* break.

Know the cost: once the manifest declares the bump, `cargo-semver-checks` reports "no semver update required" and **skips every check** for the rest of the cycle, so it catches an unintended break only in the first breaking PR of a cycle (and throughout an additive-only one). `scripts/release.sh` compensates by running the full check once at release time with `--release-type minor`, which overrides that derivation; read its output against the `[Unreleased]` section you are about to promote. An empty report on a cycle whose changelog claims a breaking change is the signal worth chasing.

`scripts/release.sh` reads the previous version from the latest `vX.Y.Z` tag, so it accepts an already-bumped manifest. It refuses a version that does not come after the last release, a manifest outside `last-release .. version-being-cut`, and a changelog a previous run already promoted.

It also refuses to release when that full check did not happen, because "no findings" and "never ran" print the same way and this is the run that has to be trusted (#337). The report runs first, before the bump and the promotion, so a stop leaves a clean tree; it stops if `cargo-semver-checks` is missing, and it stops if the tool ran but printed no `Summary` verdict — which is how a toolchain newer than the installed version fails, `cargo install cargo-semver-checks --locked` being the fix. `--skip-api-delta` releases without it and says so twice, for the window after a rustc release when no published version parses the new rustdoc format yet.

The verdict is read from that `Summary` line rather than from the exit status because the status does not mean one thing: cargo-semver-checks 0.48.0 exits 1 both for findings and for a failure to run, where 0.50.0 splits those into 100 and 101. `scripts/check-release-script.sh` pins that and the rest of the release script's behaviour — it stubs `cargo` and runs `release.sh` against a scratch clone, in about a second with no network and no build. Run it when changing `release.sh`; like `check-hdf5-18.sh` it is a command rather than a CI job, since it exercises maintainer tooling rather than the crate.

## The public API surface

```
./scripts/check-api-surface.sh
```

Every module is `pub(crate)` and the API is what `lib.rs` re-exports, which makes one mistake easy: a `pub` type in one of those modules that a re-exported signature returns or contains. A caller can hold such a value and use it — reading a `pub` field and calling an inherent method need no name — and can never write the name down. The script builds the nameable set from rustdoc JSON by walking the crate root through `pub use` and public modules, then flags every public item mentioning a type outside it. Scanning *every* item kind rather than a list of the interesting ones is both shorter and the only version that holds: a first cut looking at functions and fields alone missed a `pub const`, a `pub static`, an associated const, an associated type, and a bound on a type definition — `pub struct P<T: Unnameable>` is uninstantiable from outside and says so nowhere. `tmp/` is a fine place to inject one of each and check. It needs nightly for `--output-format json`, which is why it is a command rather than a CI job.

The one exclusion is a trait's own node, which carries its supertrait bounds: three traits here name a private `sealed::Sealed` on purpose, and refusing outside implementations is the point of that pattern. Trait *methods* are scanned like anything else.

**`cargo-semver-checks` cannot see these types**: its lints match items by importable path, so a type with no public path is invisible to all 254 lints it ships at 0.50. Measured against `v0.40.0` with two mutations of the same lint in one run: `ChunkCacheStats::index_loaded` (re-exported) `pub` -> `pub(crate)` was reported as `inherent_method_missing`, and `Superblock::serialize` (then unexported) was not reported at all. That is how `Superblock::base_address` changed from `u64` to `BaseAddress` in 0.40.0 with no finding, no changelog line, and a field consumers could see and not use.

Re-exporting is usually the fix, sealed `#[non_exhaustive]` in the same change — free while nobody can name the type to construct it or match it exhaustively — but check what else the type drags in first: `Superblock` had a `pub fn parse_from_source<S: Source + ?Sized>`, so exporting it would have made the crate-private `Source` trait a fresh leak. Its three parsing and serialization methods became `pub(crate)` for that reason.

**A re-export does not close the hole that caused the bug.** No lint fires when a public field's *type* changes; `struct_pub_field_missing` covers removal and rename only. `tests/public_api_surface.rs` is the gate for that class — it is a separate crate, so binding each `Superblock` field to its declared type there turns a retyped field into a compile error.
