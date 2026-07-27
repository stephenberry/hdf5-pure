# Project guidance

## Build hygiene

Run `cargo clean -p hdf5-pure` when builds start taking tens of seconds before any work appears to happen. Incremental compilation leaves one object file per codegen unit in `target/*/deps` on every rebuild — 256 per rebuild of the library alone — and cargo never reaps them. Cargo and rustc scan `deps/`, so once that directory holds hundreds of thousands of files the listing alone costs tens of seconds on every build. Measured here at 723k leftover objects: an incremental rebuild of the library plus one integration test took 18.3s, against 1.3s in a clean target directory.

`cargo clean -p hdf5-pure` drops this package's artifacts across every profile and feature set, including all 91 integration binaries, and leaves dependencies (and the compiled libhdf5) alone. It costs one non-incremental rebuild of the crate, about five seconds. `CARGO_INCREMENTAL=0` stops the accumulation outright, but it makes every rebuild of the library non-incremental — 4.6s against 0.9s — so the periodic clean is the better trade.

Prefer `cargo test --lib` for the fast loop: the ~580 library tests run in about two seconds, and the mutation checks that prove a new test is load-bearing belong there. `cargo test --all-features` builds 91 separate integration binaries and compiles libhdf5 from source through `hdf5-metno-src`, so treat it as a pre-push gate rather than an inner-loop command, and keep to one feature set locally — every distinct set is a full parallel copy of the build graph.

## Changelog

Keep `CHANGELOG.md` entries concise and reader-facing. Each entry is one or two sentences: lead with the user-facing capability and the public API name, keep at most one short caveat clause naming what is still refused or limited, and end with the issue/PR link in `([#NN](url))` form. Use a **Breaking:** prefix for breaking changes.

Drop the things that bloat it: bit/byte-level internals and root-cause mechanics, "validated/verified against the reference C library" narration, "byte-for-byte" boilerplate, exhaustive enumerations of every refused case, and development cross-references like "addressed below". Commit `d5a966b` (#61, "Trim the changelog to concise, reader-facing entries") is the canonical example of the before/after; match its `before -> after` for any new entry, and don't let the `[Unreleased]` section regrow into PR-description-length paragraphs.

`docs/reference/changelog.md` is a generated include of `CHANGELOG.md`, so edit only the root file.

## Versioning

The version in `Cargo.toml` is the version being *developed*, not the one last published. **A PR that breaks the public API bumps it to the next minor (`0.24.0` -> `0.25.0`), and `Cargo.lock` with it, in that same PR**, alongside its `CHANGELOG.md` entry. If it already reads the next minor, leave it: one bump covers every breaking change in the cycle. Additive-only PRs need no bump, and the release takes care of theirs.

This is what keeps the `SemVer (cargo-semver-checks)` job green. That job compares the branch against the latest crates.io release and fails a breaking change unless the manifest already declares a bump large enough to carry it. Deferring the bump to release time makes every breaking PR red by construction, and a check that is expected to be red stops being read at all, including when it flags an *unintended* break.

Know the cost: once the manifest declares the bump, `cargo-semver-checks` reports "no semver update required" and **skips every check** for the rest of the cycle, so it catches an unintended break only in the first breaking PR of a cycle (and throughout an additive-only one). `scripts/release.sh` compensates by running the full check once at release time with `--release-type minor`, which overrides that derivation; read its output against the `[Unreleased]` section you are about to promote. An empty report on a cycle whose changelog claims a breaking change is the signal worth chasing.

`scripts/release.sh` reads the previous version from the latest `vX.Y.Z` tag, so it accepts an already-bumped manifest. It refuses a version that does not come after the last release, a manifest outside `last-release .. version-being-cut`, and a changelog a previous run already promoted.
